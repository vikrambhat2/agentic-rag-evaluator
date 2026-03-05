"""Layer 2: Retrieval Quality — average relevance score across all RetrievalResults."""

from typing import List

from src.evaluator.judge import OllamaJudge, RETRIEVAL_RELEVANCE_PROMPT
from src.models.schemas import LayerScore, RetrievalResult


def score_retrieval_quality(
    query: str,
    retrieval_results: List[RetrievalResult],
    judge: OllamaJudge,
) -> LayerScore:
    """
    Score retrieval quality by calling the judge on each chunk.
    Average relevance_score across all RetrievalResults.
    Updates relevance_score on each result in-place via the judge.
    """
    if not retrieval_results:
        return LayerScore(
            layer=2,
            name="Retrieval Quality",
            score=0.0,
            details="No retrieval results",
        )

    source_scores: dict = {}

    for result in retrieval_results:
        if not result.chunks:
            source_scores[result.source] = source_scores.get(result.source, []) + [0.0]
            continue

        chunk_scores = []
        for chunk in result.chunks[:3]:  # score up to 3 chunks per source
            prompt = RETRIEVAL_RELEVANCE_PROMPT.format(query=query, chunk=chunk[:1000])
            s = judge.score(prompt)
            chunk_scores.append(s)

        avg = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        # Update the relevance_score on the result object
        result.relevance_score = avg
        source_scores[result.source] = source_scores.get(result.source, []) + [avg]

    # Build per-source breakdown
    breakdown_parts = []
    all_scores = []
    for source, scores in source_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        all_scores.append(avg)
        breakdown_parts.append(f"{source}: {avg:.2f}")

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    details = ", ".join(breakdown_parts)

    return LayerScore(
        layer=2,
        name="Retrieval Quality",
        score=round(overall_avg, 4),
        details=details,
    )
