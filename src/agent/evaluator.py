from typing import Any, Dict, List

from src.models.schemas import AgentState, RetrievalPlan, RetrievalResult


class EvaluatorNode:
    """
    LangGraph node that scores retrieved chunks and decides whether to refine.

    Uses a heuristic relevance score (chunk density) for graph routing.
    Deep LLM-judge scoring happens in the 5-layer evaluator (run_eval.py).
    Returns partial dict for LangGraph reducer compatibility.
    """

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("retrieval_plan")
        results: List[RetrievalResult] = list(state.get("retrieval_results", []))
        iteration = state.get("iteration", 0)

        if not plan or not results:
            return {"trace": ["evaluator: no results to evaluate — proceeding to generator"]}

        # Score only results from the current iteration
        scored_results = []
        for r in results:
            if r.iteration != iteration:
                scored_results.append(r)
                continue

            chunks = r.chunks or []
            if not chunks:
                score = 0.0
            else:
                meaningful = sum(1 for c in chunks if len(c.strip()) > 20)
                score = meaningful / max(len(chunks), 1)
                score = max(0.1, min(0.95, score))

            scored_results.append(RetrievalResult(
                source=r.source,
                chunks=r.chunks,
                relevance_score=score,
                iteration=r.iteration,
            ))

        # Check if refinement is needed
        current_results = [r for r in scored_results if r.iteration == iteration]
        if not current_results:
            return {
                "retrieval_results": scored_results,
                "trace": ["evaluator: proceeding to generator (no current-iteration results)"],
            }

        avg_score = sum(r.relevance_score for r in current_results) / len(current_results)
        threshold = plan.confidence_threshold
        max_iter = plan.max_iterations

        if avg_score < threshold and iteration < max_iter - 1:
            # Build refined query variants
            new_variants = [
                f"refined: {v}" for v in (plan.query_variants or [state["query"]])
            ]
            updated_plan = RetrievalPlan(
                sources=plan.sources,
                strategy=plan.strategy,
                query_variants=new_variants,
                confidence_threshold=plan.confidence_threshold,
                max_iterations=plan.max_iterations,
            )
            return {
                "retrieval_results": scored_results,
                "retrieval_plan": updated_plan,
                "iteration": iteration + 1,
                "trace": [
                    f"evaluator: avg_score={avg_score:.2f} < threshold={threshold:.2f} "
                    f"— refining (iteration {iteration + 1})"
                ],
            }
        else:
            return {
                "retrieval_results": scored_results,
                "trace": [f"evaluator: avg_score={avg_score:.2f} — proceeding to generator"],
            }


def needs_refinement(state: AgentState) -> str:
    """
    Conditional edge function after evaluator.
    Returns 'planner' if refinement is needed, else 'generator'.
    """
    plan = state.get("retrieval_plan")
    results = state.get("retrieval_results", [])
    iteration = state.get("iteration", 0)

    if not plan or not results:
        return "generator"

    # Check scores from the most recent completed iteration
    # After EvaluatorNode ran, iteration may have been incremented → look at iteration-1
    last_iter = max((r.iteration for r in results), default=0)
    current_results = [r for r in results if r.iteration == last_iter]

    if not current_results:
        return "generator"

    avg_score = sum(r.relevance_score for r in current_results) / len(current_results)
    threshold = plan.confidence_threshold
    max_iter = plan.max_iterations

    if avg_score < threshold and iteration < max_iter:
        return "planner"
    return "generator"
