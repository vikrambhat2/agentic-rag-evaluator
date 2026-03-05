"""Layer 4: Memory Efficiency — did the agent correctly use or skip cached plans?"""

from typing import List, Optional

from src.models.schemas import LayerScore, RetrievalPlan, RetrievalResult

MEMORY_THRESHOLD = 0.75


def score_memory(
    memory_hit: bool,
    memory_plan: Optional[RetrievalPlan],
    retrieval_results: List[RetrievalResult],
    judge=None,  # unused but kept for interface consistency
) -> LayerScore:
    """
    Score memory usage.

    No memory hit: score = 1.0 (fresh plan, correct behavior).
    Memory hit: score = avg_relevance / 0.75 (capped at 1.0).
      - If the cached plan produced good results, the memory helped.
      - If results were poor, the memory may have misdirected retrieval.
    """
    if not memory_hit:
        return LayerScore(
            layer=4,
            name="Memory Efficiency",
            score=1.0,
            details="No memory used — fresh plan",
        )

    # Memory was used — evaluate quality of the memory-driven retrieval
    scored = [r for r in retrieval_results if r.relevance_score > 0]
    if not scored:
        return LayerScore(
            layer=4,
            name="Memory Efficiency",
            score=0.5,
            details="Memory plan used, but no scored results available",
        )

    avg_relevance = sum(r.relevance_score for r in scored) / len(scored)
    score = min(1.0, avg_relevance / MEMORY_THRESHOLD)

    details = f"Memory plan used, avg relevance: {avg_relevance:.2f}"

    return LayerScore(
        layer=4,
        name="Memory Efficiency",
        score=round(score, 4),
        details=details,
    )
