"""
run_agent.py — Run a single query through the full LangGraph RAG pipeline.

Usage:
    python run_agent.py "your query here"
    python run_agent.py "What is retrieval-augmented generation?"
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python run_agent.py \"your query here\"[/red]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="Agentic RAG", border_style="cyan"))

    # Import here to avoid slow startup when checking args
    from src.graph.rag_graph import run_query

    console.print("\n[yellow]Running RAG graph...[/yellow]")
    state = run_query(query)

    # ── Plan ──────────────────────────────────────────────────────────────
    plan = state.get("retrieval_plan")
    if plan:
        plan_table = Table(title="Retrieval Plan", box=box.SIMPLE, show_header=True)
        plan_table.add_column("Field", style="cyan")
        plan_table.add_column("Value", style="white")
        plan_table.add_row("Sources", str(plan.sources))
        plan_table.add_row("Strategy", plan.strategy)
        plan_table.add_row("Confidence Threshold", f"{plan.confidence_threshold:.2f}")
        plan_table.add_row("Max Iterations", str(plan.max_iterations))
        plan_table.add_row("Query Variants", "\n".join(plan.query_variants))
        console.print(plan_table)
    else:
        console.print("[yellow]No retrieval plan generated.[/yellow]")

    # ── Sources Hit ────────────────────────────────────────────────────────
    results = state.get("retrieval_results", [])
    if results:
        src_table = Table(title="Sources Hit", box=box.SIMPLE)
        src_table.add_column("Source", style="green")
        src_table.add_column("Iteration", justify="center")
        src_table.add_column("Chunks Retrieved", justify="center")
        src_table.add_column("Relevance Score", justify="center")
        for r in results:
            src_table.add_row(
                r.source,
                str(r.iteration),
                str(len(r.chunks)),
                f"{r.relevance_score:.2f}",
            )
        console.print(src_table)
    else:
        console.print("[yellow]No retrieval results.[/yellow]")

    # ── Iterations ─────────────────────────────────────────────────────────
    iteration = state.get("iteration", 0)
    console.print(f"\n[cyan]Iterations taken:[/cyan] {iteration + 1}")
    console.print(f"[cyan]Memory hit:[/cyan] {state.get('memory_hit', False)}")

    # ── Final Answer ───────────────────────────────────────────────────────
    answer = state.get("final_answer", "No answer generated.")
    console.print(Panel(answer, title="[bold green]Final Answer[/bold green]", border_style="green"))

    # ── Trace ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Full Trace:[/bold cyan]")
    for i, t in enumerate(state.get("trace", []), 1):
        console.print(f"  {i:2d}. {t}")


if __name__ == "__main__":
    main()
