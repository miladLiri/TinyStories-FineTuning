"""
show_single_node_train_result.py

Reads training_metrics.json from single_node_finetuning/ and displays:
1. Run overview and hardware metadata (including Epochs planned & completed).
2. Training loss, perplexity, and token accuracy summary deltas (with Epoch progression).
3. Detailed Training Time & Efficiency Report (step duration, epoch duration, tokens/sec).
4. Dedicated Epoch-by-Epoch Progression Summary Table beside per-step data.
5. High-resolution multi-panel plots generated using Matplotlib with dual horizontal axes:
   - Bottom horizontal axis: Steps
   - Top horizontal axis: Epochs
6. Terminal ASCII convergence diagrams showing both Step and Epoch on the horizontal axis.
7. Step Progression Snapshot Table with Step and Epoch clearly juxtaposed.
"""

import json
import math
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


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration (e.g. 1h 23m 45s, 2m 15.0s, or 45.2s)."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s ({seconds:.1f}s total)"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:02.0f}s ({seconds:.1f}s total)"


def render_ascii_line_chart(values, steps, epochs=None, height=7, width=50, title="Curve", y_format="{:6.2f}"):
    """Renders a clean, terminal-friendly ASCII line chart showing both Step and Epoch on horizontal axis."""
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
    sampled_epochs = [epochs[i] for i in sampled_indices] if epochs and len(epochs) == len(steps) else None

    grid = [[" " for _ in range(num_points)] for _ in range(height)]

    for col, v in enumerate(sampled_vals):
        row = height - 1 - int(((v - min_v) / val_range) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"

    lines = []
    lines.append(f"[bold cyan]{title:^65}[/bold cyan]")
    lines.append("        ┌" + "─" * num_points + "┐")

    for r in range(height):
        label_val = max_v - (r / (height - 1)) * val_range
        label_str = y_format.format(label_val)
        row_content = "".join(grid[r])
        lines.append(f" {label_str} │[bold yellow]{row_content}[/bold yellow]│")

    lines.append("        └" + "─" * num_points + "┘")

    if sampled_epochs:
        start_s = f"Step {sampled_steps[0]} (Ep {sampled_epochs[0]:.2f})"
        end_s = f"Step {sampled_steps[-1]} (Ep {sampled_epochs[-1]:.2f})"
    else:
        start_s = f"Step {sampled_steps[0]}"
        end_s = f"Step {sampled_steps[-1]}"

    spacing = num_points - len(start_s) - len(end_s)
    if spacing > 0:
        lines.append("         " + start_s + " " * spacing + end_s)
    else:
        lines.append(f"         {start_s}  -->  {end_s}")

    return "\n".join(lines)


def generate_matplotlib_plots(data, output_path: Path):
    """Generates a high-quality 4-panel training plot using Matplotlib with dual Step & Epoch horizontal axes."""
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
    epochs = [s.get("epoch", 0.0) for s in steps_data]
    losses = [s["loss"] for s in steps_data]
    ppls = [s["perplexity"] for s in steps_data]
    accs = [s.get("accuracy", 0.0) for s in steps_data]
    speeds = [s["tokens_per_sec"] for s in steps_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")

    # Helper function to setup dual Step (bottom) and Epoch (top) horizontal axes
    def apply_dual_axes(ax):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.5, color="#D0D0D0")
        ax.tick_params(colors="#333333", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

        # Bottom axis: Step
        ax.set_xlabel("Training Step", fontsize=10, fontweight="bold", color="#333333")

        # Top axis: Epoch
        if len(steps) >= 2 and steps[-1] > steps[0]:
            s_min, s_max = float(steps[0]), float(steps[-1])
            e_min, e_max = float(epochs[0]), float(epochs[-1])

            if e_max > e_min:
                try:
                    s2e = lambda s: e_min + (s - s_min) * (e_max - e_min) / (s_max - s_min)
                    e2s = lambda e: s_min + (e - e_min) * (s_max - s_min) / (e_max - e_min)

                    secax = ax.secondary_xaxis("top", functions=(s2e, e2s))
                    secax.set_xlabel("Epoch Number", fontsize=10, fontweight="bold", color="#1D3557")
                    secax.tick_params(colors="#1D3557", labelsize=9)

                    # Mark integer epoch milestones if present
                    int_start = int(math.floor(e_min)) + 1
                    int_end = int(math.floor(e_max))
                    for ep_int in range(int_start, int_end + 1):
                        step_bound = e2s(ep_int)
                        if s_min <= step_bound <= s_max:
                            ax.axvline(step_bound, color="#6C757D", linestyle=":", linewidth=1.2, alpha=0.7)
                except Exception:
                    pass

    # 1. Training Loss
    ax_loss = axes[0, 0]
    ax_loss.plot(steps, losses, marker="o", markersize=4, color="#E63946", linewidth=2, label="Train Loss")
    ax_loss.set_title("Training Loss (Cross-Entropy)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_loss.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax_loss.legend(loc="upper right", frameon=True, facecolor="#F0F0F0")
    apply_dual_axes(ax_loss)

    # 2. Perplexity Curve
    ax_ppl = axes[0, 1]
    ax_ppl.plot(steps, ppls, marker="o", markersize=4, color="#F4A261", linewidth=2, label="Train PPL")
    ax_ppl.set_title("Perplexity [PPL = exp(loss)]", fontsize=11, fontweight="bold", color="#1D3557")
    ax_ppl.set_ylabel("Perplexity", fontsize=10, fontweight="bold")
    ax_ppl.legend(loc="upper right", frameon=True, facecolor="#F0F0F0")
    apply_dual_axes(ax_ppl)

    # 3. Training Token Accuracy (%)
    ax_acc = axes[1, 0]
    ax_acc.plot(steps, accs, marker="o", markersize=4, color="#2A9D8F", linewidth=2, label="Token Accuracy (%)")
    ax_acc.set_title("Training Token Accuracy (%)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax_acc.legend(loc="lower right", frameon=True, facecolor="#F0F0F0")
    apply_dual_axes(ax_acc)

    # 4. Training Throughput
    ax_spd = axes[1, 1]
    ax_spd.plot(steps, speeds, marker="o", markersize=4, color="#198754", linewidth=2, label="Throughput")
    avg_spd = sum(speeds) / len(speeds) if speeds else 0
    ax_spd.axhline(avg_spd, color="#0D6EFD", linestyle=":", linewidth=1.5, label=f"Avg: {avg_spd:.0f} tok/s")
    ax_spd.set_title("Training Throughput (Tokens / Second)", fontsize=11, fontweight="bold", color="#1D3557")
    ax_spd.set_ylabel("Tokens / Sec", fontsize=10, fontweight="bold")
    ax_spd.legend(loc="lower right", frameon=True, facecolor="#F0F0F0")
    apply_dual_axes(ax_spd)

    device_name = meta.get("device_name", meta.get("device", "CPU"))
    total_planned = meta.get("total_planned_steps", "N/A")
    num_planned_epochs = meta.get("num_epochs", 1)

    fig.suptitle(
        f"TinyStories-1M Training Progress (Loss, Perplexity & Accuracy)\n"
        f"Device: {device_name} | Step: {steps[-1]}/{total_planned} | Epoch: {epochs[-1]:.2f}/{num_planned_epochs} | "
        f"Final Loss: {losses[-1]:.4f} | Final Acc: {accs[-1]:.2f}%\n"
        f"[Bottom Axis: Steps  ───  Top Axis: Epochs]",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Single Node Fine-Tuning Results & Metrics Dashboard[/bold white]\n"
            "[dim]Tracking training loss, perplexity, token accuracy, epochs, time report & Matplotlib graphics[/dim]",
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
    step_count = summary.get("total_steps") or (steps_data[-1]["step"] if steps_data else 0)
    final_epoch = summary.get("total_epochs_trained") or (steps_data[-1]["epoch"] if steps_data else 0.0)
    total_planned_epochs = meta.get("num_epochs", 1)
    steps_per_epoch = meta.get("steps_per_epoch")

    # 1. Metadata Panel (with Epochs)
    meta_table = Table(box=box.ROUNDED, header_style="bold magenta", show_header=False)
    meta_table.add_column("Property", style="cyan", width=25)
    meta_table.add_column("Value", style="green")

    meta_table.add_row("Base Model", meta.get("model_name", "roneneldan/TinyStories-1M"))
    meta_table.add_row("Compute Device", f"{meta.get('device_name', 'N/A')} ({meta.get('device', 'cpu').upper()})")
    meta_table.add_row("Parameters", f"{meta.get('total_parameters', 0):,}")
    meta_table.add_row("Effective Batch Size", str(meta.get("effective_batch_size", "N/A")))
    meta_table.add_row("Peak Learning Rate", f"{meta.get('learning_rate', 0):.2e}")
    meta_table.add_row("Sequence Length", str(meta.get("max_seq_length", 256)))
    meta_table.add_row("Epoch Progress", f"{final_epoch:.2f} / {total_planned_epochs} epoch(s)")
    meta_table.add_row("Steps Completed", f"{step_count} (of {meta.get('total_planned_steps', 'N/A')} planned)")
    if steps_per_epoch:
        meta_table.add_row("Steps per Full Epoch", str(steps_per_epoch))
    meta_table.add_row("Total Training Time", format_duration(total_elapsed))
    meta_table.add_row("Total Tokens Processed", f"{total_tokens:,}")
    meta_table.add_row("Average Throughput", f"{avg_speed:.1f} tokens/s")

    console.print(Panel(meta_table, title="[bold cyan]Run Overview & Hardware Configuration[/bold cyan]", box=box.ROUNDED))
    console.print()

    if not steps_data:
        console.print("[yellow]No step metrics recorded yet. Training may have just started.[/yellow]\n")
        return

    # 2. Key Metrics Summary Table (Loss, PPL, Accuracy, and Epoch Progress)
    sum_table = Table(title="Training Summary & Metric Deltas", box=box.ROUNDED, header_style="bold magenta")
    sum_table.add_column("Metric", style="cyan")
    sum_table.add_column("Initial Value", style="yellow")
    sum_table.add_column("Final Value", style="green")
    sum_table.add_column("Improvement / Delta", style="bold magenta")

    init_epoch = steps_data[0]["epoch"]
    epoch_delta = final_epoch - init_epoch

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
        "Epoch Progress",
        f"Epoch {init_epoch:.2f}",
        f"Epoch {final_epoch:.2f}",
        f"+{epoch_delta:.2f} epochs trained",
    )
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

    # 3. Dedicated Training Time & Efficiency Report
    time_table = Table(
        title="Training Time & Efficiency Report",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_justify="left",
    )
    time_table.add_column("Time Metric", style="cyan", width=35)
    time_table.add_column("Measurement / Value", style="green")

    avg_step_time = (total_elapsed / step_count) if step_count > 0 else 0.0
    steps_per_sec = (step_count / total_elapsed) if total_elapsed > 0 else 0.0
    sec_per_1k_tokens = (1000.0 / avg_speed) if avg_speed > 0 else 0.0
    planned_steps = meta.get("total_planned_steps", 300)
    est_total_planned_time = planned_steps * avg_step_time

    # Estimate time for a full epoch
    if steps_per_epoch:
        time_per_full_epoch = avg_step_time * steps_per_epoch
    elif final_epoch > 0:
        time_per_full_epoch = total_elapsed / final_epoch
    else:
        time_per_full_epoch = 0.0

    best_step = summary.get("best_step", "N/A")
    best_epoch_val = summary.get("best_epoch", "N/A")
    best_loss_val = summary.get("best_loss", "N/A")
    best_acc_val = summary.get("best_accuracy", "N/A")

    time_table.add_row("Total Training Elapsed Time", format_duration(total_elapsed))
    time_table.add_row("Average Duration per Step", f"{avg_step_time:.3f} seconds / step")
    time_table.add_row("Average Duration per 100 Steps", format_duration(avg_step_time * 100))
    if time_per_full_epoch > 0:
        time_table.add_row("Estimated Duration per Full Epoch", format_duration(time_per_full_epoch))
    time_table.add_row("Step Frequency (Steps/Sec)", f"{steps_per_sec:.2f} steps / second")
    time_table.add_row("Token Processing Throughput", f"{avg_speed:.1f} tokens / second")
    time_table.add_row("Time per 1,000 Story Tokens", f"{sec_per_1k_tokens:.2f} seconds")

    best_location_str = f"Step {best_step}"
    if best_epoch_val != "N/A":
        best_location_str += f" (Epoch {best_epoch_val:.2f})"
    time_table.add_row("Best Model Achieved At", f"{best_location_str} [Loss: {best_loss_val}, Acc: {best_acc_val}%]")
    time_table.add_row(
        f"Projected Full Run Time ({planned_steps} Steps / {total_planned_epochs} Epoch(s))",
        format_duration(est_total_planned_time),
    )

    console.print(time_table)
    console.print()

    # 4. Dedicated Epoch Progression Summary Table beside Per-Step Data
    epoch_groups = {}
    for entry in steps_data:
        # Group by 1-based integer epoch
        ep_num = int(math.floor(entry["epoch"])) + 1
        if ep_num not in epoch_groups:
            epoch_groups[ep_num] = []
        epoch_groups[ep_num].append(entry)

    epoch_summary_table = Table(
        title="Epoch Progression Summary (Aggregated beside Step Data)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    epoch_summary_table.add_column("Epoch", style="bold green", justify="center")
    epoch_summary_table.add_column("Steps Range", style="cyan", justify="center")
    epoch_summary_table.add_column("Loss (Start -> End)", style="yellow", justify="center")
    epoch_summary_table.add_column("Best Loss", style="green", justify="right")
    epoch_summary_table.add_column("Accuracy (Start -> End)", style="bold cyan", justify="center")
    epoch_summary_table.add_column("Best Accuracy", style="bold magenta", justify="right")
    epoch_summary_table.add_column("Tokens Processed", style="white", justify="right")
    epoch_summary_table.add_column("Elapsed Time", style="dim", justify="right")

    for ep_num, ep_steps in epoch_groups.items():
        start_step = ep_steps[0]["step"]
        end_step = ep_steps[-1]["step"]
        start_loss_ep = ep_steps[0]["loss"]
        end_loss_ep = ep_steps[-1]["loss"]
        min_loss_ep = min(s["loss"] for s in ep_steps)
        start_acc_ep = ep_steps[0].get("accuracy", 0.0)
        end_acc_ep = ep_steps[-1].get("accuracy", 0.0)
        max_acc_ep = max(s.get("accuracy", 0.0) for s in ep_steps)
        ep_tokens = sum(s.get("total_tokens", 0) for s in ep_steps) // len(ep_steps)  # cumulative tokens
        ep_elapsed = ep_steps[-1]["elapsed_sec"]

        status_str = " (completed)" if final_epoch >= ep_num else f" ({ep_steps[-1]['epoch']:.2f})"
        epoch_summary_table.add_row(
            f"Epoch {ep_num}{status_str}",
            f"Steps {start_step} - {end_step}",
            f"{start_loss_ep:.4f} -> {end_loss_ep:.4f}",
            f"{min_loss_ep:.4f}",
            f"{start_acc_ep:.2f}% -> {end_acc_ep:.2f}%",
            f"{max_acc_ep:.2f}%",
            f"{ep_steps[-1]['total_tokens']:,}",
            format_duration(ep_elapsed),
        )

    console.print(epoch_summary_table)
    console.print()

    # 5. Matplotlib Graphical Plots (Dual Step & Epoch Horizontal Axes)
    console.rule("[bold cyan]Matplotlib Graphical Dashboard (Steps & Epochs)[/bold cyan]")
    with console.status("[bold green]Generating Matplotlib plots with dual Step & Epoch axes..."):
        plot_path = generate_matplotlib_plots(data, PLOT_OUTPUT_FILE)

    if plot_path and plot_path.exists():
        console.print(
            Panel(
                f"[bold green]✓ High-resolution training plots generated successfully![/bold green]\n\n"
                f"Saved to: [bold underline cyan]{plot_path.resolve()}[/bold underline cyan]\n"
                f"[dim]The plot contains 4 subplots with dual horizontal axes: Bottom = Step, Top = Epoch.[/dim]",
                title="[bold cyan]Plot Generated[/bold cyan]",
                box=box.ROUNDED,
                border_style="green",
            )
        )
        console.print()

    # 6. Terminal Visual Convergence Diagrams
    console.rule("[bold cyan]Terminal Convergence Diagrams (Step & Epoch)[/bold cyan]")
    console.print()

    steps_list = [s["step"] for s in steps_data]
    epochs_list = [s.get("epoch", 0.0) for s in steps_data]
    loss_list = [s["loss"] for s in steps_data]
    acc_list = [s.get("accuracy", 0.0) for s in steps_data]

    loss_chart = render_ascii_line_chart(
        loss_list,
        steps_list,
        epochs=epochs_list,
        height=8,
        width=50,
        title="Training Loss Curve (Lower is Better)",
        y_format="{:6.3f}",
    )
    console.print(Panel(loss_chart, box=box.ROUNDED, border_style="cyan"))
    console.print()

    acc_chart = render_ascii_line_chart(
        acc_list,
        steps_list,
        epochs=epochs_list,
        height=8,
        width=50,
        title="Training Token Accuracy Curve (%) (Higher is Better)",
        y_format="{:5.1f}%",
    )
    console.print(Panel(acc_chart, box=box.ROUNDED, border_style="cyan"))
    console.print()

    # 7. Step & Epoch History Snapshot Table
    history_table = Table(
        title=f"Step & Epoch Progression Snapshot (Last {min(10, len(steps_data))} Logged Steps)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    history_table.add_column("Step", style="cyan", justify="right")
    history_table.add_column("Epoch", style="bold yellow", justify="right")
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
            f"{entry['epoch']:.3f}",
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
