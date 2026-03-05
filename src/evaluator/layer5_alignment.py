"""Layer 5: Answer Alignment — faithfulness and relevance of the final answer."""

from typing import List

from src.evaluator.judge import (
    ANSWER_FAITHFULNESS_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    OllamaJudge,
)
from src.models.schemas import LayerScore, RetrievalResult


def score_alignment(
    query: str,
    final_answer: str,
    retrieval_results: List[RetrievalResult],
    judge: OllamaJudge,
) -> LayerScore:
    """
    Score answer alignment.

    faithfulness = judge(answer grounded in all chunks)
    relevance = judge(answer addresses query)
    score = 0.6 * faithfulness + 0.4 * relevance
    """
    # Flatten all chunks across all sources
    all_chunks = []
    for r in retrieval_results:
        all_chunks.extend(r.chunks)

    context = "\n\n---\n\n".join(all_chunks[:8])  # cap context size
    if not context:
        context = "No context retrieved."

    # Faithfulness
    faithfulness_prompt = ANSWER_FAITHFULNESS_PROMPT.format(
        query=query,
        context=context[:3000],  # avoid token limit issues
        answer=final_answer[:1000],
    )
    faithfulness = judge.score(faithfulness_prompt)

    # Relevance
    relevance_prompt = ANSWER_RELEVANCE_PROMPT.format(
        query=query,
        answer=final_answer[:1000],
    )
    relevance = judge.score(relevance_prompt)

    score = 0.6 * faithfulness + 0.4 * relevance
    details = f"Faithfulness: {faithfulness:.2f}, Relevance: {relevance:.2f}"

    return LayerScore(
        layer=5,
        name="Answer Alignment",
        score=round(score, 4),
        details=details,
    )
