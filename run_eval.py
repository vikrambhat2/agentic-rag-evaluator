"""
run_eval.py — Run the full 5-layer agentic evaluation on the test set.

Usage:
    python run_eval.py
"""

import json
import os
import sys

from rich.console import Console

console = Console()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_file = os.path.join(base_dir, "data", "test_set.json")

    # ── Load test set ──────────────────────────────────────────────────────
    if not os.path.exists(test_set_file):
        console.print(f"[red]Test set not found: {test_set_file}[/red]")
        console.print("[yellow]Run 'python ingest.py' first to generate the test set.[/yellow]")
        sys.exit(1)

    with open(test_set_file, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    console.print(f"[cyan]Loaded {len(raw_cases)} test cases from {test_set_file}[/cyan]\n")

    # Import heavy modules after startup checks
    from src.graph.rag_graph import run_query
    from src.models.schemas import (
        AgenticEvalResult,
        TestCase,
        RetrievalPlan,
    )
    from src.evaluator.judge import OllamaJudge
    from src.evaluator.layer1_plan import score_plan_quality
    from src.evaluator.layer2_retrieval import score_retrieval_quality
    from src.evaluator.layer3_refinement import score_refinement
    from src.evaluator.layer4_memory import score_memory
    from src.evaluator.layer5_alignment import score_alignment
    from src.evaluator.report import generate_report, compute_overall_score

    # Parse test cases
    test_cases = []
    for raw in raw_cases:
        try:
            tc = TestCase(**raw)
            test_cases.append(tc)
        except Exception as e:
            console.print(f"[yellow]Skipping malformed test case: {e}[/yellow]")

    if not test_cases:
        console.print("[red]No valid test cases found.[/red]")
        sys.exit(1)

    judge = OllamaJudge(model="llama3.2")
    eval_results = []

    # ── Evaluate each query ────────────────────────────────────────────────
    for i, tc in enumerate(test_cases, 1):
        console.print(f"[yellow]Evaluating {i}/{len(test_cases)}: {tc.query[:70]}...[/yellow]")

        # Run the graph
        try:
            state = run_query(tc.query)
        except Exception as e:
            console.print(f"  [red]Graph execution failed: {e}[/red]")
            continue

        plan = state.get("retrieval_plan")
        planned_sources = plan.sources if plan else []
        retrieval_results = state.get("retrieval_results", [])
        final_answer = state.get("final_answer", "")
        memory_hit = state.get("memory_hit", False)
        memory_plan = state.get("memory_plan")
        trace = state.get("trace", [])

        # Default plan for scoring if none was generated
        if plan is None:
            plan = RetrievalPlan(
                sources=["vector_db"],
                strategy="semantic",
                query_variants=[tc.query],
                confidence_threshold=0.7,
                max_iterations=1,
            )

        # ── Layer 1: Plan Quality ──────────────────────────────────────────
        try:
            l1 = score_plan_quality(
                query=tc.query,
                expected_sources=tc.expected_sources,
                planned_sources=planned_sources,
                plan=plan,
                judge=judge,
            )
        except Exception as e:
            console.print(f"  [red]L1 error: {e}[/red]")
            from src.models.schemas import LayerScore
            l1 = LayerScore(layer=1, name="Plan Quality", score=0.0, details=str(e))

        # ── Layer 2: Retrieval Quality ─────────────────────────────────────
        try:
            l2 = score_retrieval_quality(
                query=tc.query,
                retrieval_results=retrieval_results,
                judge=judge,
            )
        except Exception as e:
            console.print(f"  [red]L2 error: {e}[/red]")
            from src.models.schemas import LayerScore
            l2 = LayerScore(layer=2, name="Retrieval Quality", score=0.0, details=str(e))

        # ── Layer 3: Refinement ────────────────────────────────────────────
        try:
            l3 = score_refinement(
                retrieval_results=retrieval_results,
                trace=trace,
            )
        except Exception as e:
            console.print(f"  [red]L3 error: {e}[/red]")
            from src.models.schemas import LayerScore
            l3 = LayerScore(layer=3, name="Refinement Quality", score=0.0, details=str(e))

        # ── Layer 4: Memory ────────────────────────────────────────────────
        try:
            l4 = score_memory(
                memory_hit=memory_hit,
                memory_plan=memory_plan,
                retrieval_results=retrieval_results,
                judge=judge,
            )
        except Exception as e:
            console.print(f"  [red]L4 error: {e}[/red]")
            from src.models.schemas import LayerScore
            l4 = LayerScore(layer=4, name="Memory Efficiency", score=0.0, details=str(e))

        # ── Layer 5: Alignment ─────────────────────────────────────────────
        try:
            l5 = score_alignment(
                query=tc.query,
                final_answer=final_answer,
                retrieval_results=retrieval_results,
                judge=judge,
            )
        except Exception as e:
            console.print(f"  [red]L5 error: {e}[/red]")
            from src.models.schemas import LayerScore
            l5 = LayerScore(layer=5, name="Answer Alignment", score=0.0, details=str(e))

        layer_scores = [l1, l2, l3, l4, l5]

        # Build eval result
        eval_result = AgenticEvalResult(
            query=tc.query,
            expected_sources=tc.expected_sources,
            planned_sources=planned_sources,
            final_answer=final_answer,
            retrieval_results=retrieval_results,
            memory_hit=memory_hit,
            trace=trace,
            layer_scores=layer_scores,
            overall_score=0.0,  # computed next
        )
        eval_result.overall_score = compute_overall_score(eval_result)

        eval_results.append(eval_result)

        # Brief per-query summary
        console.print(
            f"  L1={l1.score:.2f} | L2={l2.score:.2f} | L3={l3.score:.2f} | "
            f"L4={l4.score:.2f} | L5={l5.score:.2f} | "
            f"Overall={eval_result.overall_score:.2f}"
        )

    # ── Generate and print report ──────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]GENERATING EVALUATION REPORT[/bold cyan]")
    console.print("=" * 60 + "\n")

    generate_report(eval_results)


if __name__ == "__main__":
    main()
