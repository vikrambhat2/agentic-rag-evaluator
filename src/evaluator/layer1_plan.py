"""Layer 1: Plan Quality — Jaccard source overlap + LLM judge."""

from typing import List

from src.evaluator.judge import OllamaJudge, PLAN_QUALITY_PROMPT
from src.models.schemas import LayerScore, RetrievalPlan, SourceType


def score_plan_quality(
    query: str,
    expected_sources: List[SourceType],
    planned_sources: List[SourceType],
    plan: RetrievalPlan,
    judge: OllamaJudge,
) -> LayerScore:
    """
    Score the retrieval plan quality.

    Hard score: Jaccard similarity between expected and planned sources.
    LLM score: judge evaluates appropriateness of the plan.
    Final = 0.5 * jaccard + 0.5 * llm_score
    """
    # Jaccard similarity
    expected_set = set(expected_sources)
    planned_set = set(planned_sources)
    intersection = expected_set & planned_set
    union = expected_set | planned_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # LLM judge score
    prompt = PLAN_QUALITY_PROMPT.format(
        query=query,
        planned_sources=planned_sources,
        strategy=plan.strategy,
    )
    llm_score = judge.score(prompt)

    final_score = 0.5 * jaccard + 0.5 * llm_score

    details = (
        f"Expected {expected_sources}, Got {planned_sources}, "
        f"Jaccard={jaccard:.2f}, LLM={llm_score:.2f}"
    )

    return LayerScore(
        layer=1,
        name="Plan Quality",
        score=round(final_score, 4),
        details=details,
    )
