"""
train_single_node.py

Fine-tunes TinyStories-1M on instruction-based prompts using Hugging Face Transformers Trainer & TrainingArguments.
Hardcodes all hyperparameters at the top of the file.
Tracks training loss, perplexity, and token accuracy.
Only saves the SINGLE BEST checkpoint (lowest loss / highest accuracy) to disk.
Auto-detects CUDA / CPU and records comprehensive LLM metrics on-the-go to JSON.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_callback import PrinterCallback

# UTF-8 console output guard for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# HARDCODED HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
# Paths
MODEL_PATH = "./artifacts/model"
TRAIN_DATA_PATH = "./fine-tuning-dataset/train.json"
OUTPUT_DIR = "./single_node_finetuning"
BEST_CHECKPOINT_DIR = "./single_node_finetuning/best_checkpoint"
METRICS_FILE = "./single_node_finetuning/training_metrics.json"

# Optimization & Batching
BATCH_SIZE = 16                   # Batch size per step (utilizes GTX 1650 Ti VRAM safely, ~200MB)
GRAD_ACCUM_STEPS = 2             # Effective batch size = 32 (reduces gradient variance)
LEARNING_RATE = 8e-5             # Calibrated fine-tuning learning rate (prevents weight thrashing)
WEIGHT_DECAY = 0.01              # Weight decay for regularization
MAX_SEQ_LENGTH = 288             # Max sequence length (captures >95% of full stories)

# Steps & Epochs
NUM_EPOCHS = 6                   # Number of training epochs
MAX_STEPS = None                 # Set to None for full epoch training (or integer to limit steps)
WARMUP_STEPS = 60                # Gentle linear warmup steps

# Logging intervals
LOG_EVERY_STEPS = 5              # Log metrics and check best checkpoint every N steps

# Dataset limits
MAX_TRAIN_SAMPLES = 10000        # Training subset size (expanded from 5,000 for higher accuracy)
SEED = 42                        # Random seed for reproducibility
# ==============================================================================


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InstructionStoryDataset(Dataset):
    """
    Consumes pre-formatted items from fine-tuning-dataset/:
    {
      "prompt": "Instruction: Write a short story using the words: [words].\nStory:\n",
      "story": "[story text]"
    }
    """

    def __init__(self, items, tokenizer, max_length=256):
        self.examples = []
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for item in items:
            prompt = item.get("prompt", "")
            story = item.get("story", "").strip()
            if not story or not prompt:
                continue

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            story_ids = tokenizer.encode(story + tokenizer.eos_token, add_special_tokens=False)

            # Combine prompt and story
            input_ids = prompt_ids + story_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            # Label masking: set prompt tokens to -100 so loss and accuracy are only computed on story tokens
            labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(input_ids))
            for i in range(prompt_len):
                labels[i] = -100

            attention_mask = [1] * len(input_ids)

            # Pad to max_length
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len
                attention_mask += [0] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def save_metrics_to_disk(metrics_data, path):
    """Save metrics dictionary to JSON file on disk."""
    temp_path = str(path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    os.replace(temp_path, str(path))


class TinyStoriesTrainer(Trainer):
    """
    Subclasses Hugging Face Trainer to collect token-level accuracy
    and count processed tokens during training forward passes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_window = []
        self.acc_window = []
        self.total_tokens_processed = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if model.training:
            with torch.no_grad():
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs["labels"][..., 1:].contiguous()
                mask = shift_labels != -100
                if mask.sum() > 0:
                    preds = shift_logits.argmax(dim=-1)
                    batch_acc = (preds[mask] == shift_labels[mask]).float().mean().item() * 100.0
                else:
                    batch_acc = 0.0

                batch_tokens = (inputs["attention_mask"] == 1).sum().item()
                self.total_tokens_processed += batch_tokens
                self.loss_window.append(loss.item())
                self.acc_window.append(batch_acc)

        return (loss, outputs) if return_outputs else loss


class MetricsTrackingCallback(TrainerCallback):
    """
    Callback for Hugging Face Trainer:
    1. Tracks loss, perplexity, token accuracy, speed, and time every LOG_EVERY_STEPS.
    2. Saves ONLY the single best checkpoint to single_node_finetuning/best_checkpoint.
    3. Saves metrics on-the-go to single_node_finetuning/training_metrics.json.
    """

    def __init__(
        self,
        metrics_data,
        metrics_file,
        best_ckpt_path,
        start_time,
        total_steps,
        num_epochs,
        log_every_steps,
        device_type,
    ):
        self.metrics_data = metrics_data
        self.metrics_file = metrics_file
        self.best_ckpt_path = Path(best_ckpt_path)
        self.start_time = start_time
        self.total_steps = total_steps
        self.num_epochs = num_epochs
        self.log_every_steps = log_every_steps
        self.device_type = device_type

        self.last_logged_step = 0
        self.best_loss = float("inf")
        self.best_step = 0
        self.best_epoch = 0.0
        self.best_acc = 0.0
        self.trainer = None

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step % self.log_every_steps == 0 and state.global_step > self.last_logged_step:
            trainer = self.trainer or kwargs.get("trainer")
            cur_time = time.time()
            elapsed = cur_time - self.start_time

            loss_window = trainer.loss_window if trainer else []
            acc_window = trainer.acc_window if trainer else []
            total_tokens = trainer.total_tokens_processed if trainer else 0

            avg_loss = sum(loss_window) / len(loss_window) if loss_window else 0.0
            avg_acc = sum(acc_window) / len(acc_window) if acc_window else 0.0
            if loss_window:
                loss_window.clear()
            if acc_window:
                acc_window.clear()

            perplexity = math.exp(min(avg_loss, 20.0))

            current_lr = 0.0
            if trainer and trainer.lr_scheduler:
                current_lr = trainer.lr_scheduler.get_last_lr()[0]
            elif trainer and trainer.optimizer and trainer.optimizer.param_groups:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
            elif state.log_history:
                current_lr = state.log_history[-1].get("learning_rate", 0.0)

            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

            vram_mb = 0.0
            if self.device_type == "cuda":
                vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

            cur_epoch = round(state.epoch, 3) if state.epoch is not None else 0.0

            step_metric = {
                "step": state.global_step,
                "epoch": cur_epoch,
                "loss": round(avg_loss, 4),
                "perplexity": round(perplexity, 2),
                "accuracy": round(avg_acc, 2),
                "learning_rate": float(f"{current_lr:.6e}"),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "total_tokens": total_tokens,
                "elapsed_sec": round(elapsed, 2),
                "vram_allocated_mb": round(vram_mb, 1),
            }
            self.metrics_data["steps"].append(step_metric)

            # Check and save ONLY the single best checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
                self.best_acc = avg_acc
                self.best_step = state.global_step
                self.best_epoch = cur_epoch
                self.best_ckpt_path.mkdir(parents=True, exist_ok=True)
                target_model = model or (trainer.model if trainer else None)
                target_tokenizer = tokenizer or (trainer.processing_class if trainer else None)
                if target_model:
                    target_model.save_pretrained(str(self.best_ckpt_path))
                if target_tokenizer:
                    target_tokenizer.save_pretrained(str(self.best_ckpt_path))

            best_flag = " [*BEST*]" if is_best else ""
            print(
                f"Step {state.global_step:4d}/{self.total_steps} (Ep {cur_epoch:.2f}/{self.num_epochs}) | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {perplexity:6.2f} | "
                f"Acc: {avg_acc:5.2f}% | "
                f"Speed: {tokens_per_sec:5.0f} tok/s | "
                f"Time: {elapsed:5.1f}s{best_flag}",
                flush=True,
            )

            # Write metrics to JSON on the go
            save_metrics_to_disk(self.metrics_data, self.metrics_file)
            self.last_logged_step = state.global_step


def main():
    set_seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = Path(BEST_CHECKPOINT_DIR)

    # 1. Device Selection (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" Single Node Fine-Tuning: TinyStories-1M (HF Trainer API) ")
    print("=" * 65)
    print(f"[*] Target Device:       {device.type.upper()}")
    if device.type == "cuda":
        print(f"[*] GPU Model:           {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"[*] Output Directory:    {out_dir.resolve()}")
    print(f"[*] Best Checkpoint:     {best_ckpt_path.resolve()} (single best only)")
    print(f"[*] Metrics Output:      {Path(METRICS_FILE).resolve()}")
    print("=" * 65)

    # 2. Load Tokenizer & Model
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        print(f"[bold red]Error:[/bold red] Base model not found in '{model_dir.resolve()}'.")
        print("[yellow]Please run 'python download_artifacts.py' first to download the model.[/yellow]\n")
        return

    print(f"[*] Loading tokenizer & model from '{MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(str(model_dir.resolve()))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model loaded. Total parameters: {total_params:,}")

    # 3. Load Prepared Training Dataset
    print(f"[*] Loading prepared training data from '{TRAIN_DATA_PATH}'...")
    if not Path(TRAIN_DATA_PATH).exists():
        print(f"[!] Prepared dataset not found at '{TRAIN_DATA_PATH}'. Please run 'python download_artifacts.py' first.")
        return

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_raw = json.load(f)

    if MAX_TRAIN_SAMPLES is not None:
        train_raw = train_raw[:MAX_TRAIN_SAMPLES]

    print(f"[*] Tokenizing {len(train_raw):,} training samples...")
    train_dataset = InstructionStoryDataset(train_raw, tokenizer, max_length=MAX_SEQ_LENGTH)

    steps_per_epoch = (len(train_dataset) // BATCH_SIZE) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    if MAX_STEPS is not None:
        total_steps = min(total_steps, MAX_STEPS)

    # 4. Metrics Storage Initialization
    metrics_data = {
        "metadata": {
            "model_name": "roneneldan/TinyStories-1M",
            "device": device.type,
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "total_parameters": total_params,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "total_planned_steps": total_steps,
            "num_epochs": NUM_EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "steps": [],
        "summary": {},
    }

    start_time = time.time()

    # 5. Hugging Face TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS is not None else -1,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=LOG_EVERY_STEPS,
        seed=SEED,
        max_grad_norm=1.0,
        save_strategy="no",                 # Do not save multiple checkpoint-N dirs
        report_to="none",                   # Disable wandb/tensorboard
        disable_tqdm=True,                  # Keep terminal logs clean and formatted
        use_cpu=(device.type == "cpu"),     # Automatically select CPU or CUDA
        dataloader_drop_last=False,
    )

    # 6. Callback Initialization
    callback = MetricsTrackingCallback(
        metrics_data=metrics_data,
        metrics_file=METRICS_FILE,
        best_ckpt_path=best_ckpt_path,
        start_time=start_time,
        total_steps=total_steps,
        num_epochs=NUM_EPOCHS,
        log_every_steps=LOG_EVERY_STEPS,
        device_type=device.type,
    )

    # 7. Trainer Initialization
    trainer = TinyStoriesTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    callback.trainer = trainer
    trainer.remove_callback(PrinterCallback)

    print("\n" + "=" * 65)
    print(f"[*] Starting Hugging Face Trainer: {total_steps} planned steps ({NUM_EPOCHS} epoch(s))...")
    print("=" * 65)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C). Saving current state...")

    # 8. Final Summary & Metrics Save
    total_elapsed = time.time() - start_time
    final_epoch = metrics_data["steps"][-1]["epoch"] if metrics_data["steps"] else 0.0
    final_step = metrics_data["steps"][-1]["step"] if metrics_data["steps"] else 0

    print("\n" + "=" * 65)
    print(f"[*] Training completed {final_step} steps ({final_epoch:.2f} epochs) in {total_elapsed:.2f}s.")
    print(f"[*] Best checkpoint achieved at Step {callback.best_step} (Epoch {callback.best_epoch:.2f}) [Loss: {callback.best_loss:.4f}, Acc: {callback.best_acc:.2f}%].")
    print(f"[*] Best checkpoint stored at: '{best_ckpt_path.resolve()}'")

    first_loss = metrics_data["steps"][0]["loss"] if metrics_data["steps"] else None
    last_loss = metrics_data["steps"][-1]["loss"] if metrics_data["steps"] else None
    first_ppl = metrics_data["steps"][0]["perplexity"] if metrics_data["steps"] else None
    last_ppl = metrics_data["steps"][-1]["perplexity"] if metrics_data["steps"] else None
    first_acc = metrics_data["steps"][0]["accuracy"] if metrics_data["steps"] else None
    last_acc = metrics_data["steps"][-1]["accuracy"] if metrics_data["steps"] else None

    metrics_data["summary"] = {
        "status": "completed" if (MAX_STEPS and final_step >= MAX_STEPS) else "finished",
        "total_steps": final_step,
        "total_epochs_trained": final_epoch,
        "total_elapsed_sec": round(total_elapsed, 2),
        "total_tokens_processed": trainer.total_tokens_processed,
        "average_tokens_per_sec": round(trainer.total_tokens_processed / total_elapsed, 1) if total_elapsed > 0 else 0,
        "initial_loss": first_loss,
        "final_loss": last_loss,
        "initial_perplexity": first_ppl,
        "final_perplexity": last_ppl,
        "initial_accuracy": first_acc,
        "final_accuracy": last_acc,
        "best_step": callback.best_step,
        "best_epoch": callback.best_epoch,
        "best_loss": round(callback.best_loss, 4) if callback.best_loss != float("inf") else last_loss,
        "best_accuracy": round(callback.best_acc, 2) if callback.best_loss != float("inf") else last_acc,
        "loss_reduction": round(first_loss - last_loss, 4) if (first_loss and last_loss) else 0,
        "accuracy_gain": round(last_acc - first_acc, 2) if (first_acc is not None and last_acc is not None) else 0,
    }

    save_metrics_to_disk(metrics_data, METRICS_FILE)
    print(f"[✓] Best checkpoint and final metrics saved to '{OUTPUT_DIR}'.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
