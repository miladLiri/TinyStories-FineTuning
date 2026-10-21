"""
train_single_node.py

Fine-tunes TinyStories-1M on instruction-based prompts using Hugging Face Transformers.
Hardcodes all hyperparameters at the top of the file.
Uses training set samples to track training loss, perplexity, and token accuracy.
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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

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
METRICS_FILE = "./single_node_finetuning/training_metrics.json"

# Optimization & Batching
BATCH_SIZE = 8                    # Batch size per step (low VRAM usage: <1GB)
GRAD_ACCUM_STEPS = 2              # Effective batch size = 16
LEARNING_RATE = 3e-4              # Initial learning rate for AdamW
WEIGHT_DECAY = 0.01               # Weight decay for regularization
MAX_SEQ_LENGTH = 256              # Max sequence length (tokens)

# Steps & Epochs
NUM_EPOCHS = 1                    # Number of training epochs
MAX_STEPS = 300                   # Maximum steps (set to None for full epoch)
WARMUP_STEPS = 20                 # Linear warmup steps

# Logging & Checkpointing intervals
LOG_EVERY_STEPS = 5               # Log metrics every N steps
SAVE_EVERY_STEPS = 50             # Save model checkpoint every N steps

# Dataset limits
MAX_TRAIN_SAMPLES = 5000          # Training subset size
SEED = 42                         # Random seed for reproducibility
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


def main():
    set_seed(SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device Selection (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" Single Node Fine-Tuning: TinyStories-1M Instruction Alignment ")
    print("=" * 65)
    print(f"[*] Target Device:       {device.type.upper()}")
    if device.type == "cuda":
        print(f"[*] GPU Model:           {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"[*] Output Directory:    {out_dir.resolve()}")
    print(f"[*] Metrics Output:      {Path(METRICS_FILE).resolve()}")
    print("=" * 65)

    # 2. Load Tokenizer & Model
    print(f"[*] Loading tokenizer & model from '{MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.train()

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    if MAX_STEPS is not None:
        total_steps = min(total_steps, MAX_STEPS)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    # 5. Metrics Storage Initialization
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
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "steps": [],
        "summary": {},
    }

    print("\n" + "=" * 65)
    print(f"[*] Starting training loop: {total_steps} planned steps...")
    print("=" * 65)

    step_counter = 0
    total_tokens_processed = 0
    loss_window = []
    acc_window = []
    start_time = time.time()
    last_log_time = start_time

    try:
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss / GRAD_ACCUM_STEPS

                # Backward pass
                loss.backward()

                # Compute training token accuracy on unmasked labels
                with torch.no_grad():
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    mask = shift_labels != -100
                    if mask.sum() > 0:
                        preds = shift_logits.argmax(dim=-1)
                        batch_acc = (preds[mask] == shift_labels[mask]).float().mean().item() * 100.0
                    else:
                        batch_acc = 0.0

                loss_window.append(outputs.loss.item())
                acc_window.append(batch_acc)

                # Count non-padded tokens
                batch_tokens = (attention_mask == 1).sum().item()
                total_tokens_processed += batch_tokens

                # Optimizer step upon gradient accumulation
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_counter += 1

                    # Logging
                    if step_counter % LOG_EVERY_STEPS == 0:
                        cur_time = time.time()
                        elapsed = cur_time - start_time
                        last_log_time = cur_time

                        avg_loss = sum(loss_window) / len(loss_window)
                        avg_acc = sum(acc_window) / len(acc_window)
                        loss_window.clear()
                        acc_window.clear()

                        perplexity = math.exp(min(avg_loss, 20.0))
                        current_lr = scheduler.get_last_lr()[0]
                        tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0

                        # Memory stats
                        vram_mb = 0.0
                        if device.type == "cuda":
                            vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

                        step_metric = {
                            "step": step_counter,
                            "epoch": round(epoch + (batch_idx + 1) / len(train_loader), 3),
                            "loss": round(avg_loss, 4),
                            "perplexity": round(perplexity, 2),
                            "accuracy": round(avg_acc, 2),
                            "learning_rate": float(f"{current_lr:.6e}"),
                            "tokens_per_sec": round(tokens_per_sec, 1),
                            "total_tokens": total_tokens_processed,
                            "elapsed_sec": round(elapsed, 2),
                            "vram_allocated_mb": round(vram_mb, 1),
                        }
                        metrics_data["steps"].append(step_metric)

                        print(
                            f"Step {step_counter:4d}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {perplexity:6.2f} | "
                            f"Acc: {avg_acc:5.2f}% | "
                            f"LR: {current_lr:.2e} | "
                            f"Speed: {tokens_per_sec:5.0f} tok/s | "
                            f"Time: {elapsed:5.1f}s"
                        )

                        # Write metrics to JSON on the go
                        save_metrics_to_disk(metrics_data, METRICS_FILE)

                    # Periodic Checkpoint Saving
                    if step_counter % SAVE_EVERY_STEPS == 0:
                        ckpt_dir = out_dir / f"checkpoint-{step_counter}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(ckpt_dir))
                        tokenizer.save_pretrained(str(ckpt_dir))
                        print(f"  [+] Saved checkpoint to {ckpt_dir.name}")

                    if MAX_STEPS is not None and step_counter >= MAX_STEPS:
                        break

            if MAX_STEPS is not None and step_counter >= MAX_STEPS:
                break

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C). Saving current state...")

    # Final Save
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"[*] Training finished in {total_elapsed:.2f}s ({step_counter} steps completed).")
    print(f"[*] Saving final model & tokenizer to '{OUTPUT_DIR}'...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save summary
    first_loss = metrics_data["steps"][0]["loss"] if metrics_data["steps"] else None
    last_loss = metrics_data["steps"][-1]["loss"] if metrics_data["steps"] else None
    first_ppl = metrics_data["steps"][0]["perplexity"] if metrics_data["steps"] else None
    last_ppl = metrics_data["steps"][-1]["perplexity"] if metrics_data["steps"] else None
    first_acc = metrics_data["steps"][0]["accuracy"] if metrics_data["steps"] else None
    last_acc = metrics_data["steps"][-1]["accuracy"] if metrics_data["steps"] else None

    metrics_data["summary"] = {
        "status": "completed" if (MAX_STEPS and step_counter >= MAX_STEPS) else "finished",
        "total_steps": step_counter,
        "total_elapsed_sec": round(total_elapsed, 2),
        "total_tokens_processed": total_tokens_processed,
        "average_tokens_per_sec": round(total_tokens_processed / total_elapsed, 1) if total_elapsed > 0 else 0,
        "initial_loss": first_loss,
        "final_loss": last_loss,
        "initial_perplexity": first_ppl,
        "final_perplexity": last_ppl,
        "initial_accuracy": first_acc,
        "final_accuracy": last_acc,
        "loss_reduction": round(first_loss - last_loss, 4) if (first_loss and last_loss) else 0,
        "accuracy_gain": round(last_acc - first_acc, 2) if (first_acc is not None and last_acc is not None) else 0,
    }

    save_metrics_to_disk(metrics_data, METRICS_FILE)
    print(f"[✓] Final model and metrics successfully saved to '{OUTPUT_DIR}'.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
