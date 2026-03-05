"""
OllamaJudge: LLM-as-judge for RAG evaluation.
All judge calls use temperature=0 for reproducibility.
"""

import re

from langchain_ollama import ChatOllama

# ─────────────────────────────────────────────
# Judge prompts
# ─────────────────────────────────────────────

RETRIEVAL_RELEVANCE_PROMPT = """You are an evaluation judge.
Given a query and a retrieved context chunk, score how relevant the chunk is to the query.
Score from 0.0 (completely irrelevant) to 1.0 (highly relevant and directly answers the query).
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Query: {query}
Context chunk: {chunk}
Score:"""

ANSWER_FAITHFULNESS_PROMPT = """You are an evaluation judge.
Given a query, all retrieved context chunks, and a generated answer, score how faithful the answer is to the context.
A faithful answer only uses information present in the context — no hallucination.
Score from 0.0 (completely hallucinated) to 1.0 (fully grounded in context).
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Query: {query}
Context: {context}
Answer: {answer}
Score:"""

ANSWER_RELEVANCE_PROMPT = """You are an evaluation judge.
Given a query and a generated answer, score how well the answer addresses the query.
Score from 0.0 (completely off-topic) to 1.0 (directly and completely answers the query).
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Query: {query}
Answer: {answer}
Score:"""

PLAN_QUALITY_PROMPT = """You are an evaluation judge for an agentic RAG system.
Available sources: vector_db (AI/ML research documentation), filesystem (internal notes and experiments), api (latest AI news, model releases, benchmarks).
Given a query and the retrieval plan that was generated, score the plan quality.
A good plan selects the right sources for the query and uses an appropriate strategy.
Score from 0.0 (completely wrong sources/strategy) to 1.0 (optimal plan for the query).
Return ONLY a decimal number between 0.0 and 1.0. Nothing else.

Query: {query}
Planned sources: {planned_sources}
Strategy: {strategy}
Score:"""


# ─────────────────────────────────────────────
# OllamaJudge class
# ─────────────────────────────────────────────

class OllamaJudge:
    """Wraps ChatOllama as an LLM judge that returns a float score 0.0–1.0."""

    def __init__(self, model: str = "llama3.2"):
        self.llm = ChatOllama(model=model, temperature=0)

    def score(self, prompt: str) -> float:
        """
        Send the prompt to the judge LLM and parse a float score.
        Returns 0.0 on any failure.
        """
        try:
            response = self.llm.invoke([("human", prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_score(raw)
        except Exception as e:
            print(f"[OllamaJudge] Error: {e}")
            return 0.0

    def _parse_score(self, text: str) -> float:
        """Extract the first float in [0, 1] from the response."""
        text = text.strip()
        # Try direct float parse
        try:
            val = float(text)
            return max(0.0, min(1.0, val))
        except ValueError:
            pass

        # Try regex extraction
        matches = re.findall(r"\b(0?\.\d+|1\.0|0|1)\b", text)
        if matches:
            try:
                val = float(matches[0])
                return max(0.0, min(1.0, val))
            except ValueError:
                pass

        print(f"[OllamaJudge] Could not parse score from: {text!r}")
        return 0.0
