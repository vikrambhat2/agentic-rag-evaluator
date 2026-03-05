"""Layer 3: Refinement Quality — did iterative refinement improve results?"""

from typing import List

from src.models.schemas import LayerScore, RetrievalResult


def score_refinement(
    retrieval_results: List[RetrievalResult],
    trace: List[str],
) -> LayerScore:
    """
    Score the agent's refinement behavior.

    - 1 iteration: score = 1.0 (no refinement needed — got it right first time)
    - Multiple iterations: score = improvement_ratio (final / first, capped at 1.0)
    - If scores got worse: score = 0.2
    """
    if not retrieval_results:
        return LayerScore(
            layer=3,
            name="Refinement Quality",
            score=1.0,
            details="No retrieval results — trivially perfect",
        )

    # Find all unique iterations
    iterations = sorted(set(r.iteration for r in retrieval_results))
    n_iterations = len(iterations)

    if n_iterations <= 1:
        return LayerScore(
            layer=3,
            name="Refinement Quality",
            score=1.0,
            details=f"Iterations: 1, Score delta: +0.00",
        )

    # Compute average relevance per iteration
    iter_scores = {}
    for it in iterations:
        iter_results = [r for r in retrieval_results if r.iteration == it]
        if iter_results:
            avg = sum(r.relevance_score for r in iter_results) / len(iter_results)
            iter_scores[it] = avg

    first_score = iter_scores.get(iterations[0], 0.0)
    final_score = iter_scores.get(iterations[-1], 0.0)
    delta = final_score - first_score

    if first_score <= 0.0:
        # Can't divide by zero; treat as no improvement
        score = 0.5
    elif final_score >= first_score:
        # Improvement or same
        ratio = final_score / first_score
        score = min(1.0, ratio)
    else:
        # Scores got worse
        score = 0.2

    details = f"Iterations: {n_iterations}, Score delta: {delta:+.2f}"

    return LayerScore(
        layer=3,
        name="Refinement Quality",
        score=round(score, 4),
        details=details,
    )
