from src.models.schemas import AgentState, RetrievalResult


class EvaluatorNode:
    """
    LangGraph node that scores retrieved chunks and decides whether to refine.
    Scoring is deferred to the OllamaJudge in the 5-layer evaluator;
    here we use a simple heuristic for the graph control flow.
    """

    def __call__(self, state: AgentState) -> AgentState:
        plan = state.get("retrieval_plan")
        results: list = state.get("retrieval_results", [])
        iteration = state.get("iteration", 0)

        if not plan or not results:
            state["trace"].append("evaluator: no results to evaluate — proceeding to generator")
            return state

        # Score each result: fraction of chunks that are non-empty
        scored_results = []
        for r in results:
            if r.iteration != iteration:
                # Keep previous iterations unchanged
                scored_results.append(r)
                continue

            chunks = r.chunks or []
            if not chunks:
                score = 0.0
            else:
                # Heuristic: score based on chunk density (non-empty, length > 20)
                meaningful = sum(1 for c in chunks if len(c.strip()) > 20)
                score = meaningful / max(len(chunks), 1)
                # Clamp to [0.1, 0.95] so it stays realistic
                score = max(0.1, min(0.95, score))

            updated = RetrievalResult(
                source=r.source,
                chunks=r.chunks,
                relevance_score=score,
                iteration=r.iteration,
            )
            scored_results.append(updated)

        state["retrieval_results"] = scored_results

        # Determine if refinement is needed
        current_results = [r for r in scored_results if r.iteration == iteration]
        if not current_results:
            state["trace"].append("evaluator: proceeding to generator (no current-iteration results)")
            return state

        avg_score = sum(r.relevance_score for r in current_results) / len(current_results)
        threshold = plan.confidence_threshold
        max_iter = plan.max_iterations

        if avg_score < threshold and iteration < max_iter - 1:
            # Signal refinement needed
            state["iteration"] = iteration + 1
            # Update query_variants for the refined pass
            new_variants = []
            for variant in (plan.query_variants or [state["query"]]):
                new_variants.append(f"refined: {variant}")
            plan.query_variants = new_variants
            state["retrieval_plan"] = plan
            state["trace"].append(
                f"evaluator: avg_score={avg_score:.2f} < threshold={threshold:.2f} "
                f"— refining (iteration {state['iteration']})"
            )
        else:
            state["trace"].append(
                f"evaluator: avg_score={avg_score:.2f} — proceeding to generator"
            )

        return state


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

    current_results = [r for r in results if r.iteration == iteration - 1]
    if not current_results:
        return "generator"

    avg_score = sum(r.relevance_score for r in current_results) / len(current_results)
    threshold = plan.confidence_threshold
    max_iter = plan.max_iterations

    # We already incremented iteration in EvaluatorNode; check if we still have room
    if avg_score < threshold and iteration < max_iter:
        return "planner"
    return "generator"
