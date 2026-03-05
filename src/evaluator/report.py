"""Report generation: rich table + JSON export for the agentic evaluation results."""

import json
import os
from typing import List

from rich.console import Console
from rich.table import Table
from rich import box

from src.models.schemas import AgenticEvalReport, AgenticEvalResult

console = Console()

WEAK_SPOT_THRESHOLD = 0.70

# Layer weights (must sum to 1.0)
LAYER_WEIGHTS = {
    1: 0.20,  # Plan Quality
    2: 0.25,  # Retrieval Quality
    3: 0.20,  # Refinement
    4: 0.15,  # Memory
    5: 0.20,  # Alignment
}


def compute_overall_score(result: AgenticEvalResult) -> float:
    """Compute weighted overall score from 5 layer scores."""
    total = 0.0
    for ls in result.layer_scores:
        weight = LAYER_WEIGHTS.get(ls.layer, 0.0)
        total += weight * ls.score
    return round(total, 4)


def generate_report(results: List[AgenticEvalResult]) -> AgenticEvalReport:
    """Build AgenticEvalReport, print rich table, flag weak spots, save JSON."""

    if not results:
        console.print("[red]No evaluation results to report.[/red]")
        return AgenticEvalReport(
            results=[],
            avg_layer1_plan=0.0,
            avg_layer2_retrieval=0.0,
            avg_layer3_refinement=0.0,
            avg_layer4_memory=0.0,
            avg_layer5_alignment=0.0,
            avg_overall=0.0,
        )

    # ── Build Rich table ──────────────────────────────────────────────────
    table = Table(
        title="Agentic RAG Evaluation Report",
        box=box.ROUNDED,
        show_footer=True,
        footer_style="bold cyan",
    )

    table.add_column("Query", style="white", max_width=40, no_wrap=False)
    table.add_column("L1 Plan", justify="center", style="cyan")
    table.add_column("L2 Retrieval", justify="center", style="green")
    table.add_column("L3 Refine", justify="center", style="yellow")
    table.add_column("L4 Memory", justify="center", style="magenta")
    table.add_column("L5 Align", justify="center", style="blue")
    table.add_column("Overall", justify="center", style="bold white")

    layer_sums = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    overall_sum = 0.0

    def _get_layer_score(result: AgenticEvalResult, layer_num: int) -> float:
        for ls in result.layer_scores:
            if ls.layer == layer_num:
                return ls.score
        return 0.0

    def _fmt(score: float) -> str:
        color = "green" if score >= 0.7 else ("yellow" if score >= 0.5 else "red")
        return f"[{color}]{score:.2f}[/{color}]"

    for r in results:
        l1 = _get_layer_score(r, 1)
        l2 = _get_layer_score(r, 2)
        l3 = _get_layer_score(r, 3)
        l4 = _get_layer_score(r, 4)
        l5 = _get_layer_score(r, 5)
        ov = r.overall_score

        layer_sums[1] += l1
        layer_sums[2] += l2
        layer_sums[3] += l3
        layer_sums[4] += l4
        layer_sums[5] += l5
        overall_sum += ov

        query_short = r.query[:38] + ".." if len(r.query) > 40 else r.query

        table.add_row(
            query_short,
            _fmt(l1),
            _fmt(l2),
            _fmt(l3),
            _fmt(l4),
            _fmt(l5),
            _fmt(ov),
        )

    n = len(results)
    avg_l1 = layer_sums[1] / n
    avg_l2 = layer_sums[2] / n
    avg_l3 = layer_sums[3] / n
    avg_l4 = layer_sums[4] / n
    avg_l5 = layer_sums[5] / n
    avg_overall = overall_sum / n

    console.print(table)

    # ── Per-layer averages ─────────────────────────────────────────────────
    console.print("\n[bold cyan]Per-Layer Averages:[/bold cyan]")
    console.print(f"  L1 Plan Quality:      {avg_l1:.3f}")
    console.print(f"  L2 Retrieval Quality: {avg_l2:.3f}")
    console.print(f"  L3 Refinement:        {avg_l3:.3f}")
    console.print(f"  L4 Memory Efficiency: {avg_l4:.3f}")
    console.print(f"  L5 Answer Alignment:  {avg_l5:.3f}")
    console.print(f"  [bold]Overall:              {avg_overall:.3f}[/bold]")

    # ── Weak spot detection ────────────────────────────────────────────────
    console.print("\n[bold yellow]Weak Spot Detection (threshold < 0.70):[/bold yellow]")
    layer_names = {1: "Plan Quality", 2: "Retrieval Quality", 3: "Refinement",
                   4: "Memory Efficiency", 5: "Answer Alignment"}
    avgs = {1: avg_l1, 2: avg_l2, 3: avg_l3, 4: avg_l4, 5: avg_l5}
    any_weak = False
    for layer_num, avg in avgs.items():
        if avg < WEAK_SPOT_THRESHOLD:
            console.print(
                f"  [red]WARNING: Layer {layer_num} ({layer_names[layer_num]}) "
                f"avg = {avg:.3f} — below threshold[/red]"
            )
            any_weak = True
    if not any_weak:
        console.print("  [green]All layers above threshold. System performing well.[/green]")

    # ── Worst query trace ─────────────────────────────────────────────────
    worst = min(results, key=lambda r: r.overall_score)
    console.print(f"\n[bold red]Worst-Performing Query (score={worst.overall_score:.3f}):[/bold red]")
    console.print(f"  Query: {worst.query}")
    console.print("  Trace:")
    for t in worst.trace:
        console.print(f"    → {t}")

    # ── Build report object ────────────────────────────────────────────────
    report = AgenticEvalReport(
        results=results,
        avg_layer1_plan=round(avg_l1, 4),
        avg_layer2_retrieval=round(avg_l2, 4),
        avg_layer3_refinement=round(avg_l3, 4),
        avg_layer4_memory=round(avg_l4, 4),
        avg_layer5_alignment=round(avg_l5, 4),
        avg_overall=round(avg_overall, 4),
    )

    # ── Save JSON report ───────────────────────────────────────────────────
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report_path = os.path.join(base, "agentic_eval_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        console.print(f"\n[green]Full report saved to: {report_path}[/green]")
    except Exception as e:
        console.print(f"\n[red]Error saving report: {e}[/red]")

    return report
