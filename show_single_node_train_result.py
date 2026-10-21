"""
show_single_node_train_result.py

Reads training_metrics.json from single_node_finetuning/
and displays:
1. Formatted terminal overview tables with training loss, perplexity, and token accuracy.
2. High-resolution multi-panel plots generated using matplotlib (Loss, PPL, Accuracy, Throughput).
3. Terminal ASCII line diagrams for loss and accuracy.
"""

import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# UTF-8 console output guard for Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

METRICS_FILE = Path("./single_node_finetuning/training_metrics.json")
PLOT_OUTPUT_FILE = Path("./single_node_finetuning/training_metrics_plot.png")


def render_ascii_line_chart(values, steps, height=7, width=50, title="Curve", y_format="{:6.2f}"):
    """Renders a clean, terminal-friendly ASCII line chart for time-series metrics."""
    if not values or len(values) < 2:
        return f"[dim]Not enough data points ({len(values)}) to render chart.[/dim]"

    min_v = min(values)
    max_v = max(values)
    val_range = max_v - min_v if max_v != min_v else 1.0

    num_points = min(len(values), width)
    stride = (len(values) - 1) / (num_points - 1) if num_points > 1 else 1
    sampled_indices = [int(round(i * stride)) for i in range(num_points)]
    sampled_vals = [values[i] for i in sampled_indices]
    sampled_steps = [steps[i] for i in sampled_indices]

    grid = [[" " for _ in range(num_points)] for _ in range(height)]

    for col, v in enumerate(sampled_vals):
        row = height - 1 - int(((v - min_v) / val_range) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    lines = []
    lines.append(f"[bold cyan]{title:^60}[/bold cyan]")
    lines.append("        ┌" + "─" * num_points + "┐")

    for r in range(height):
        label_val = max_v - (r / (height - 1)) * val_range
        label_str = y_format.format(label_val)
        row_content = "".join(grid[r])
        lines.append(f" {label_str} │[bold yellow]{row_content}[/bold yellow]│")

    lines.append("        └" + "─" * num_points + "┘")
    start_s = f"Step {sampled_steps[0]}"
    end_s = f"Step {sampled_steps[-1]}"
    spacing = num_points - len(start_s) - len(end_s)
    if spacing > 0:
        lines.append("         " + start_s + " " * spacing + end_s)

    return "\n".join(lines)


def generate_matplotlib_plots(data, output_path: Path):
    """Generates a high-quality 4-panel training plot using Matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[bold red]Matplotlib not found.[/bold red] Please install it using: pip install matplotlib\n")
        return None

    meta = data.get("metadata", {})
    steps_data = data.get("steps", [])

    if not steps_data:
        return None

    steps = [s["step"] for s in steps_data]
    losses = [s["loss"] for s in steps_data]
    ppls = [s["perplexity"] for s in steps_data]
    accs = [s.get("accuracy", 0.0) for s in steps_data]
    speeds = [s["tokens_per_sec"] for s in steps_data]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")

    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.5, color="#D0D0D0")
        ax.tick_params(colors="#333333", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    # 1. Training Loss
    ax_loss = axes[0, 0]
    ax_loss.plot(steps, losses, marker="o", markersize=4, color="#E63946", linewidth=2, label="Train Loss")
    ax_loss.set_title("Training Loss (Cross-Entropy)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_loss.set_xlabel("Step", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.legend(loc="upper right", frameon=True, facecolor="#F0F0F0")

    # 2. Perplexity Curve
    ax_ppl = axes[0, 1]
    ax_ppl.plot(steps, ppls, marker="o", markersize=4, color="#F4A261", linewidth=2, label="Train PPL")
    ax_ppl.set_title("Perplexity [PPL = exp(loss)]", fontsize=11, fontweight="bold", color="#1D3557")
    ax_ppl.set_xlabel("Step", fontsize=10)
    ax_ppl.set_ylabel("Perplexity", fontsize=10)
    ax_ppl.legend(loc="upper right", frameon=True, facecolor="#F0F0F0")

    # 3. Training Token Accuracy (%)
    ax_acc = axes[1, 0]
    ax_acc.plot(steps, accs, marker="o", markersize=4, color="#2A9D8F", linewidth=2, label="Token Accuracy (%)")
    ax_acc.set_title("Training Token Accuracy (%)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_acc.set_xlabel("Step", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.legend(loc="lower right", frameon=True, facecolor="#F0F0F0")

    # 4. Training Throughput
    ax_spd = axes[1, 1]
    ax_spd.plot(steps, speeds, marker="o", markersize=4, color="#198754", linewidth=2, label="Throughput")
    avg_spd = sum(speeds) / len(speeds) if speeds else 0
    ax_spd.axhline(avg_spd, color="#0D6EFD", linestyle=":", linewidth=1.5, label=f"Avg: {avg_spd:.0f} tok/s")
    ax_spd.set_title("Training Throughput (Tokens / Second)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_spd.set_xlabel("Step", fontsize=10)
    ax_spd.set_ylabel("Tokens / Sec", fontsize=10)
    ax_spd.legend(loc="lower right", frameon=True, facecolor="#F0F0F0")

    device_name = meta.get("device_name", meta.get("device", "CPU"))
    fig.suptitle(
        f"TinyStories-1M Training Progress (Loss, Perplexity & Accuracy)\n"
        f"Device: {device_name} | Batch Size: {meta.get('effective_batch_size', 'N/A')} | "
        f"Final Loss: {losses[-1]:.4f} | Final Acc: {accs[-1]:.2f}%",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Single Node Fine-Tuning Results & Metrics Dashboard[/bold white]\n"
            "[dim]Tracking training loss, perplexity, token accuracy & Matplotlib graphics[/dim]",
            border_style="magenta",
            box=box.DOUBLE,
        )
    )
    console.print()

    if not METRICS_FILE.exists():
        console.print(f"[bold red]Error:[/bold red] Metrics file '{METRICS_FILE}' not found.")
        console.print("[yellow]Please run 'python train_single_node.py' first to generate metrics.[/yellow]\n")
        return

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    steps_data = data.get("steps", [])
    summary = data.get("summary", {})

    total_elapsed = summary.get("total_elapsed_sec") or (steps_data[-1]["elapsed_sec"] if steps_data else 0.0)
    total_tokens = summary.get("total_tokens_processed") or (steps_data[-1]["total_tokens"] if steps_data else 0)
    avg_speed = summary.get("average_tokens_per_sec") or (round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0.0)

    # 1. Metadata Panel
    meta_table = Table(box=box.ROUNDED, header_style="bold magenta", show_header=False)
    meta_table.add_column("Property", style="cyan", width=22)
    meta_table.add_column("Value", style="green")

    meta_table.add_row("Base Model", meta.get("model_name", "N/A"))
    meta_table.add_row("Compute Device", f"{meta.get('device_name', 'N/A')} ({meta.get('device', 'cpu').upper()})")
    meta_table.add_row("Parameters", f"{meta.get('total_parameters', 0):,}")
    meta_table.add_row("Effective Batch Size", str(meta.get("effective_batch_size", "N/A")))
    meta_table.add_row("Peak Learning Rate", f"{meta.get('learning_rate', 0):.2e}")
    meta_table.add_row("Sequence Length", str(meta.get("max_seq_length", 256)))
    meta_table.add_row("Total Steps Recorded", str(len(steps_data)))
    meta_table.add_row("Elapsed Training Time", f"{total_elapsed:.2f}s")
    meta_table.add_row("Total Tokens Processed", f"{total_tokens:,}")
    meta_table.add_row("Average Throughput", f"{avg_speed:.1f} tokens/s")

    console.print(Panel(meta_table, title="[bold cyan]Run Overview & Hardware Configuration[/bold cyan]", box=box.ROUNDED))
    console.print()

    if not steps_data:
        console.print("[yellow]No step metrics recorded yet. Training may have just started.[/yellow]\n")
        return

    # 2. Key Metrics Summary Table (Loss, PPL, and Accuracy)
    sum_table = Table(title="Training Summary & Metric Deltas", box=box.ROUNDED, header_style="bold magenta")
    sum_table.add_column("Metric", style="cyan")
    sum_table.add_column("Initial Value", style="yellow")
    sum_table.add_column("Final Value", style="green")
    sum_table.add_column("Improvement / Delta", style="bold magenta")

    init_loss = summary.get("initial_loss") or steps_data[0]["loss"]
    final_loss = summary.get("final_loss") or steps_data[-1]["loss"]
    loss_delta = final_loss - init_loss

    init_ppl = summary.get("initial_perplexity") or steps_data[0]["perplexity"]
    final_ppl = summary.get("final_perplexity") or steps_data[-1]["perplexity"]
    ppl_delta = final_ppl - init_ppl

    init_acc = summary.get("initial_accuracy") or steps_data[0].get("accuracy", 0.0)
    final_acc = summary.get("final_accuracy") or steps_data[-1].get("accuracy", 0.0)
    acc_delta = final_acc - init_acc

    sum_table.add_row(
        "Training Loss (Cross-Entropy)",
        f"{init_loss:.4f}",
        f"{final_loss:.4f}",
        f"{loss_delta:+.4f} ({loss_delta / init_loss * 100:+.1f}%)" if init_loss else "N/A",
    )
    sum_table.add_row(
        "Perplexity (PPL)",
        f"{init_ppl:.2f}",
        f"{final_ppl:.2f}",
        f"{ppl_delta:+.2f} ({ppl_delta / init_ppl * 100:+.1f}%)" if init_ppl else "N/A",
    )
    sum_table.add_row(
        "Training Token Accuracy",
        f"{init_acc:.2f}%",
        f"{final_acc:.2f}%",
        f"{acc_delta:+.2f}%",
    )

    console.print(sum_table)
    console.print()

    # 3. Matplotlib Graphical Plots
    console.rule("[bold cyan]Matplotlib Graphical Dashboard[/bold cyan]")
    with console.status("[bold green]Generating Matplotlib plots..."):
        plot_path = generate_matplotlib_plots(data, PLOT_OUTPUT_FILE)

    if plot_path and plot_path.exists():
        console.print(
            Panel(
                f"[bold green]✓ High-resolution training plots generated successfully![/bold green]\n\n"
                f"Saved to: [bold underline cyan]{plot_path.resolve()}[/bold underline cyan]\n"
                f"[dim]The plot contains 4 subplots: Training Loss, Perplexity, Token Accuracy (%), and Throughput.[/dim]",
                title="[bold cyan]Plot Generated[/bold cyan]",
                box=box.ROUNDED,
                border_style="green",
            )
        )
        console.print()

    # 4. Terminal Visual Diagrams
    console.rule("[bold cyan]Terminal Convergence Diagrams[/bold cyan]")
    console.print()

    steps_list = [s["step"] for s in steps_data]
    loss_list = [s["loss"] for s in steps_data]
    acc_list = [s.get("accuracy", 0.0) for s in steps_data]

    loss_chart = render_ascii_line_chart(
        loss_list,
        steps_list,
        height=8,
        width=48,
        title="Training Loss Curve (Lower is Better)",
        y_format="{:6.3f}",
    )
    console.print(Panel(loss_chart, box=box.ROUNDED, border_style="cyan"))
    console.print()

    acc_chart = render_ascii_line_chart(
        acc_list,
        steps_list,
        height=8,
        width=48,
        title="Training Token Accuracy Curve (%) (Higher is Better)",
        y_format="{:5.1f}%",
    )
    console.print(Panel(acc_chart, box=box.ROUNDED, border_style="cyan"))
    console.print()

    # 5. Step History Snapshot Table
    history_table = Table(
        title=f"Step Progression Snapshot (Last {min(10, len(steps_data))} Logged Steps)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    history_table.add_column("Step", style="cyan", justify="right")
    history_table.add_column("Epoch", style="dim", justify="right")
    history_table.add_column("Loss", style="yellow", justify="right")
    history_table.add_column("Perplexity", style="green", justify="right")
    history_table.add_column("Accuracy", style="bold cyan", justify="right")
    history_table.add_column("Learning Rate", style="white", justify="right")
    history_table.add_column("Throughput", style="magenta", justify="right")
    history_table.add_column("Elapsed", style="dim", justify="right")
    if meta.get("device") == "cuda":
        history_table.add_column("VRAM", style="blue", justify="right")

    for entry in steps_data[-10:]:
        row = [
            str(entry["step"]),
            f"{entry['epoch']:.2f}",
            f"{entry['loss']:.4f}",
            f"{entry['perplexity']:.2f}",
            f"{entry.get('accuracy', 0.0):.2f}%",
            f"{entry['learning_rate']:.2e}",
            f"{entry['tokens_per_sec']:.0f} tok/s",
            f"{entry['elapsed_sec']:.1f}s",
        ]
        if meta.get("device") == "cuda":
            row.append(f"{entry.get('vram_allocated_mb', 0):.0f} MB")
        history_table.add_row(*row)

    console.print(history_table)
    console.print()


if __name__ == "__main__":
    main()
