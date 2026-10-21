import random
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# UTF-8 encoding safeguard for Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

# Configuration paths
ARTIFACTS_DIR = Path("./artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"
DATASET_DIR = ARTIFACTS_DIR / "dataset"


def format_size(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def get_dir_size(path: Path) -> int:
    """Calculate total size of files in a directory."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def inspect_dataset():
    """Inspect dataset size, structure, splits, and display 2 random samples."""
    console.rule("[bold cyan]1. Dataset Inspection[/bold cyan]")

    if not DATASET_DIR.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset directory '{DATASET_DIR}' not found.")
        console.print("[yellow]Please run 'python download_artifacts.py' first.[/yellow]\n")
        return None

    disk_size = get_dir_size(DATASET_DIR)
    json_files = sorted(list(DATASET_DIR.glob("*.json")))

    # Check if native JSON files exist
    if json_files:
        import json

        overview_table = Table(
            title="Native JSON Dataset Overview",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_justify="left",
        )
        overview_table.add_column("Property", style="cyan", no_wrap=True)
        overview_table.add_column("Value", style="green")

        overview_table.add_row("Location", str(DATASET_DIR.resolve()))
        overview_table.add_row("Format", "Native JSON files (.json)")
        overview_table.add_row("Disk Usage", format_size(disk_size))
        overview_table.add_row("JSON Files Found", f"{len(json_files)} files ({json_files[0].name} ... {json_files[-1].name})")

        with console.status("[bold green]Inspecting sample JSON file..."):
            with open(json_files[0], "r", encoding="utf-8") as f:
                first_file_data = json.load(f)
            stories_per_file = len(first_file_data)
            est_total = stories_per_file * len(json_files)

        overview_table.add_row("Stories per File", f"~{stories_per_file:,}")
        overview_table.add_row("Total Stories (Est.)", f"~{est_total:,} stories")
        console.print(overview_table)
        console.print()

        # Schema Table
        schema_table = Table(
            title="JSON Object Schema & Structure",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_justify="left",
        )
        schema_table.add_column("Field Name", style="cyan", no_wrap=True)
        schema_table.add_column("Data Type", style="yellow")
        schema_table.add_column("Description", style="white")

        sample_obj = first_file_data[0]
        schema_table.add_row("story", "string", "Full story text")
        schema_table.add_row("instruction", "dict", "Prompt, target words, and features used for generation")
        schema_table.add_row("summary", "string", "Short summary or moral of the story")
        schema_table.add_row("source", "string", "Source model (e.g., GPT-3.5 or GPT-4)")
        console.print(schema_table)
        console.print()

        # 1 Sample JSON Item
        console.print("[bold yellow]Dataset Sample (Single Item from Native JSON List):[/bold yellow]")
        random_file = random.choice(json_files)
        with open(random_file, "r", encoding="utf-8") as f:
            pool = json.load(f)
        sample_item = random.choice(pool)

        story_text = sample_item.get("story", "").strip()
        summary = sample_item.get("summary", "N/A")
        source = sample_item.get("source", "N/A")
        instruction = sample_item.get("instruction", {})
        prompt_instruction = instruction.get("prompt:", instruction.get("prompt", "N/A"))
        words = instruction.get("words", [])
        features = instruction.get("features", [])

        # Display formatted JSON item table
        item_table = Table(
            title=f"Sample JSON Item Details (from {random_file.name})",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_justify="left",
        )
        item_table.add_column("Key", style="cyan", width=14, no_wrap=True)
        item_table.add_column("Value / Content", style="white")

        item_table.add_row("source", source)
        item_table.add_row("target words", ", ".join(words) if words else "None specified")
        item_table.add_row("features", ", ".join(features) if features else "None")
        item_table.add_row("instruction", prompt_instruction.strip())
        item_table.add_row("summary", summary)
        item_table.add_row("story", story_text)

        console.print(item_table)
        console.print()

        return pool

    from datasets import load_from_disk

    with console.status("[bold green]Loading dataset from disk..."):
        dataset = load_from_disk(str(DATASET_DIR))

    # 1. Dataset Overview Table
    overview_table = Table(
        title="Dataset Overview",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_justify="left",
    )
    overview_table.add_column("Property", style="cyan", no_wrap=True)
    overview_table.add_column("Value", style="green")

    overview_table.add_row("Location", str(DATASET_DIR.resolve()))
    overview_table.add_row("Disk Usage", format_size(disk_size))

    total_rows = 0
    if hasattr(dataset, "keys"):
        splits = list(dataset.keys())
        overview_table.add_row("Splits Found", ", ".join(splits))
        for split_name in splits:
            split_len = len(dataset[split_name])
            total_rows += split_len
            overview_table.add_row(f"  Split: '{split_name}'", f"{split_len:,} rows")
    else:
        total_rows = len(dataset)
        overview_table.add_row("Rows", f"{total_rows:,} rows")

    overview_table.add_row("Total Rows", f"{total_rows:,} rows")
    console.print(overview_table)
    console.print()

    # 2. Dataset Structure Table
    schema_table = Table(
        title="Dataset Schema & Structure",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_justify="left",
    )
    schema_table.add_column("Column Name", style="cyan", no_wrap=True)
    schema_table.add_column("Data Type", style="yellow")
    schema_table.add_column("Description", style="white")

    # Inspect features from first split or root
    features = dataset[splits[0]].features if hasattr(dataset, "keys") else dataset.features
    for col_name, feature_type in features.items():
        desc = "Story content (synthetically generated)" if col_name == "text" else "Metadata"
        schema_table.add_row(col_name, str(feature_type), desc)

    console.print(schema_table)
    console.print()

    # 3. One Sample Item
    console.print("[bold yellow]Dataset Sample (Single Item from Dataset):[/bold yellow]")
    sample_pool = dataset[splits[0]] if hasattr(dataset, "keys") else dataset
    sample_idx = random.randint(0, len(sample_pool) - 1)
    sample_item = sample_pool[sample_idx]
    story_text = sample_item.get("text", "").strip()
    word_count = len(story_text.split())
    char_count = len(story_text)

    panel_content = (
        f"[dim]Index in split: {sample_idx} | Words: {word_count} | Characters: {char_count}[/dim]\n\n"
        f"{story_text}"
    )
    console.print(
        Panel(
            panel_content,
            title="[bold green]Sample Item[/bold green]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )
    console.print()

    return dataset


def inspect_model_and_tokenizer():
    """Inspect model architecture, parameters, and tokenizer specs."""
    console.rule("[bold cyan]2. Model & Tokenizer Specifications[/bold cyan]")

    if not MODEL_DIR.exists():
        console.print(f"[bold red]Error:[/bold red] Model directory '{MODEL_DIR}' not found.")
        console.print("[yellow]Please run 'python download_artifacts.py' first.[/yellow]\n")
        return None, None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    with console.status("[bold green]Loading tokenizer and model from disk..."):
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))

    model_disk_size = get_dir_size(MODEL_DIR)
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. Model Specs Table
    model_table = Table(
        title="Model Specifications (TinyStories-1M)",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_justify="left",
    )
    model_table.add_column("Property", style="cyan", no_wrap=True)
    model_table.add_column("Specification", style="green")

    model_table.add_row("Model Architecture", getattr(config, "architectures", ["Unknown"])[0])
    model_table.add_row("Model Type", getattr(config, "model_type", "Unknown"))
    model_table.add_row("Total Parameters", f"{num_params:,}")
    model_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    model_table.add_row("Hidden Size (d_model)", str(getattr(config, "hidden_size", "N/A")))
    model_table.add_row("Layers (num_layers)", str(getattr(config, "num_layers", "N/A")))
    model_table.add_row("Attention Heads", str(getattr(config, "num_heads", "N/A")))
    model_table.add_row("Context Window", str(getattr(config, "max_position_embeddings", "N/A")))
    model_table.add_row("Activation Function", str(getattr(config, "activation_function", "N/A")))
    model_table.add_row("Vocab Size", f"{getattr(config, 'vocab_size', 'N/A'):,}")
    model_table.add_row("Weight Precision (dtype)", str(next(model.parameters()).dtype))
    model_table.add_row("Model Disk Size", format_size(model_disk_size))

    console.print(model_table)
    console.print()

    # 2. Tokenizer Specs Table
    tok_table = Table(
        title="Tokenizer Specifications",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_justify="left",
    )
    tok_table.add_column("Property", style="cyan", no_wrap=True)
    tok_table.add_column("Specification", style="green")

    tok_table.add_row("Tokenizer Class", tokenizer.__class__.__name__)
    tok_table.add_row("Vocabulary Size", f"{len(tokenizer):,}")
    tok_table.add_row("BOS Token", f"{tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    tok_table.add_row("EOS Token", f"{tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    tok_table.add_row("PAD Token", f"{tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    tok_table.add_row("UNK Token", f"{tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

    console.print(tok_table)
    console.print()

    return model, tokenizer


def run_inference(model, tokenizer):
    """Run inference with instruction-based prompts and display formatted baseline results."""
    console.rule("[bold cyan]3. Model Inference & Story Generation (Instruction Baseline)[/bold cyan]")

    if model is None or tokenizer is None:
        console.print("[bold red]Cannot run inference: Model or Tokenizer not loaded.[/bold red]\n")
        return

    import torch

    instruction_tasks = [
        {
            "prompt": "Instruction: Write a short story using the words: frog, pond, jump.\nStory:",
            "constraints": "Words to include: frog, pond, jump",
        },
        {
            "prompt": "Instruction: Write a short story about sharing. The story should use the words: cookie, friend, happy.\nStory:",
            "constraints": "Theme: sharing | Words: cookie, friend, happy",
        },
    ]

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i, task in enumerate(instruction_tasks, 1):
        prompt = task["prompt"]
        constraints = task["constraints"]
        console.print(f"[bold yellow]Running Instruction Prompt #{i}...[/bold yellow]")

        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )

        elapsed = time.time() - start_time
        new_tokens_count = output_tokens.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = new_tokens_count / elapsed if elapsed > 0 else 0

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Extract only the newly generated continuation
        completion = generated_text[len(prompt):].strip()

        # Display generation result in a table
        res_table = Table(
            title=f"Instruction Test #{i} (Base Model Baseline)",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_justify="left",
            show_header=True,
        )
        res_table.add_column("Field", style="cyan", width=20)
        res_table.add_column("Content / Value", style="white")

        res_table.add_row("Instruction Prompt", f"[bold green]{prompt}[/bold green]")
        res_table.add_row("Target Constraints", f"[yellow]{constraints}[/yellow]")
        res_table.add_row("Base Model Output", completion)
        res_table.add_row(
            "Performance",
            f"{new_tokens_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/s)",
        )
        res_table.add_row(
            "Baseline Observation",
            "[dim yellow]Pre-fine-tuning baseline: The base completion model generates text without adhering strictly to the instruction or requested words. Fine-tuning on the dataset's instruction pairs will teach it to follow these constraints.[/dim yellow]",
        )

        console.print(res_table)
        console.print()


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold white]TinyStories Artifacts Inspector & Inference Test[/bold white]\n"
            "[dim]Validating dataset structure, model parameters, and text generation[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )
    console.print()

    # Step 1: Inspect dataset
    inspect_dataset()

    # Step 2: Inspect model & tokenizer
    model, tokenizer = inspect_model_and_tokenizer()

    # Step 3: Run inference
    if model is not None and tokenizer is not None:
        run_inference(model, tokenizer)


if __name__ == "__main__":
    main()
