"""
test_single_node_result_artifact.py

Compares the Base TinyStories-1M model against the Fine-Tuned model.
Loads both models and runs 2 sample instruction prompts side-by-side,
evaluating instruction compliance and target word inclusion.
"""

import sys
import time
from pathlib import Path

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

# UTF-8 console output guard for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

BASE_MODEL_PATH = Path("./artifacts/model")
FT_BEST_CHECKPOINT_PATH = Path("./single_node_finetuning/best_checkpoint")
FT_FALLBACK_PATH = Path("./single_node_finetuning")


def check_word_presence(text: str, target_words: list):
    """Check how many target words appear in the generated story."""
    text_lower = text.lower()
    found = [w for w in target_words if w.lower() in text_lower]
    return found, len(found) == len(target_words)


def generate_story(model, tokenizer, prompt: str, device, seed: int = 42, max_new_tokens: int = 90):
    """Generate completion using consistent sampling parameters."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip()
    new_tokens = output_tokens.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    return completion, tok_per_sec


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold white]TinyStories-1M: Base vs. Fine-Tuned Model Comparison[/bold white]\n"
            "[dim]Evaluating instruction-following ability and target word adherence[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )
    console.print()

    # 1. Verify Model Paths
    if not BASE_MODEL_PATH.exists():
        console.print(f"[bold red]Error:[/bold red] Base model directory '{BASE_MODEL_PATH}' not found.")
        console.print("[yellow]Please run 'python download_artifacts.py' first.[/yellow]\n")
        return

    ft_path = FT_BEST_CHECKPOINT_PATH if FT_BEST_CHECKPOINT_PATH.exists() else FT_FALLBACK_PATH
    if not ft_path.exists() or not (ft_path / "model.safetensors").exists() and not (ft_path / "pytorch_model.bin").exists():
        console.print(f"[bold red]Error:[/bold red] Fine-tuned model not found in '{ft_path}'.")
        console.print("[yellow]Please run 'python train_single_node.py' first to produce a fine-tuned model.[/yellow]\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Models & Tokenizer
    with console.status("[bold green]Loading Base Model and Fine-Tuned Model..."):
        tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_PATH)).to(device)
        base_model.eval()

        ft_model = AutoModelForCausalLM.from_pretrained(str(ft_path)).to(device)
        ft_model.eval()

    console.print(f"[bold cyan]Device:[/bold cyan] {device.type.upper()}")
    console.print(f"[bold cyan]Base Model Path:[/bold cyan] {BASE_MODEL_PATH.resolve()}")
    console.print(f"[bold cyan]Fine-Tuned Path:[/bold cyan] {ft_path.resolve()}\n")

    # 3. Two Sample Instruction Prompts
    test_prompts = [
        {
            "id": 1,
            "instruction": "Instruction: Write a short story using the words: frog, pond, jump.\nStory:\n",
            "target_words": ["frog", "pond", "jump"],
        },
        {
            "id": 2,
            "instruction": "Instruction: Write a short story using the words: cookie, friend, happy.\nStory:\n",
            "target_words": ["cookie", "friend", "happy"],
        },
    ]

    summary_rows = []

    for test in test_prompts:
        prompt_id = test["id"]
        prompt_text = test["instruction"]
        target_words = test["target_words"]

        console.rule(f"[bold cyan]Test Prompt #{prompt_id}[/bold cyan]")
        console.print(f"[bold yellow]Prompt Text:[/bold yellow]\n{prompt_text}")
        console.print(f"[bold yellow]Required Words:[/bold yellow] [magenta]{', '.join(target_words)}[/magenta]\n")

        with console.status(f"[bold green]Generating outputs for Prompt #{prompt_id}..."):
            base_output, base_speed = generate_story(base_model, tokenizer, prompt_text, device)
            ft_output, ft_speed = generate_story(ft_model, tokenizer, prompt_text, device)

        base_found, base_ok = check_word_presence(base_output, target_words)
        ft_found, ft_ok = check_word_presence(ft_output, target_words)

        # Build comparison table
        comp_table = Table(
            title=f"Prompt #{prompt_id} Model Outputs Comparison",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_header=True,
        )
        comp_table.add_column("Attribute", style="cyan", width=22)
        comp_table.add_column("Base Model (Pre-trained)", style="white", width=45)
        comp_table.add_column("Fine-Tuned Model (Post-trained)", style="white", width=45)

        comp_table.add_row(
            "Target Words Found",
            f"{len(base_found)}/{len(target_words)} ({', '.join(base_found) if base_found else 'none'})",
            f"{len(ft_found)}/{len(target_words)} ({', '.join(ft_found) if ft_found else 'none'})",
        )
        comp_table.add_row(
            "Adherence Status",
            "[green]Complete[/green]" if base_ok else "[red]Missing words[/red]",
            "[green]Complete[/green]" if ft_ok else "[yellow]Partial / Missing[/yellow]",
        )
        comp_table.add_row(
            "Generated Story",
            base_output,
            ft_output,
        )
        comp_table.add_row(
            "Generation Speed",
            f"{base_speed:.1f} tokens/s",
            f"{ft_speed:.1f} tokens/s",
        )

        console.print(comp_table)
        console.print()

        summary_rows.append((
            f"Prompt #{prompt_id}",
            ", ".join(target_words),
            f"{len(base_found)}/{len(target_words)}",
            f"{len(ft_found)}/{len(target_words)}",
            "[green]+ Improved[/green]" if len(ft_found) > len(base_found) else (
                "[blue]= Equal[/blue]" if len(ft_found) == len(base_found) else "[yellow]Needs more steps[/yellow]"
            ),
        ))

    # 4. Final Comparison Summary Table
    final_table = Table(title="Overall Adherence & Accuracy Summary", box=box.ROUNDED, header_style="bold magenta")
    final_table.add_column("Test Case", style="cyan")
    final_table.add_column("Required Words", style="white")
    final_table.add_column("Base Model Score", style="yellow", justify="center")
    final_table.add_column("Fine-Tuned Model Score", style="green", justify="center")
    final_table.add_column("Result", style="bold magenta", justify="center")

    for row in summary_rows:
        final_table.add_row(*row)

    console.print(final_table)
    console.print()


if __name__ == "__main__":
    main()
