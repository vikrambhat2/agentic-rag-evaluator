import json
import re
from typing import Any, Dict

from langchain_ollama import ChatOllama

from src.models.schemas import AgentState, RetrievalPlan

DEFAULT_PLAN = RetrievalPlan(
    sources=["vector_db"],
    strategy="semantic",
    query_variants=[""],
    confidence_threshold=0.7,
    max_iterations=1,
)

PLANNER_SYSTEM_PROMPT = """You are a retrieval planner for an agentic RAG system.
Available sources: vector_db (AI/ML research docs), filesystem (internal notes), api (latest AI news and benchmarks).
Given the query, decide which sources to retrieve from, what strategy to use, and generate query variants for each source.
Return only JSON matching this schema: {sources: [], strategy: str, query_variants: [], confidence_threshold: float, max_iterations: int}
max_iterations must be between 1 and 3. confidence_threshold between 0.5 and 0.9."""


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _parse_plan(raw: str, query: str) -> RetrievalPlan:
    """Parse LLM JSON output into a RetrievalPlan. Falls back to default on failure."""
    try:
        cleaned = _strip_markdown(raw)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        data = json.loads(match.group())

        sources = data.get("sources", ["vector_db"])
        valid_sources = [s for s in sources if s in ("vector_db", "filesystem", "api")]
        if not valid_sources:
            valid_sources = ["vector_db"]

        strategy = data.get("strategy", "semantic")
        query_variants = data.get("query_variants", [query])
        if not query_variants:
            query_variants = [query]

        confidence_threshold = float(data.get("confidence_threshold", 0.7))
        confidence_threshold = max(0.5, min(0.9, confidence_threshold))

        max_iterations = int(data.get("max_iterations", 1))
        max_iterations = max(1, min(3, max_iterations))

        return RetrievalPlan(
            sources=valid_sources,
            strategy=strategy,
            query_variants=query_variants,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
        )
    except Exception as e:
        print(f"[Planner] JSON parse error: {e}. Using default plan.")
        plan = DEFAULT_PLAN.model_copy()
        plan.query_variants = [query]
        return plan


class PlannerNode:
    """LangGraph node that generates a RetrievalPlan for the current query."""

    def __init__(self, model: str = "llama3.2"):
        self.llm = ChatOllama(model=model, temperature=0.1)

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Return partial dict update (not full state) for LangGraph reducer compatibility."""
        query = state["query"]

        # Reuse memory plan if available
        if state.get("memory_hit") and state.get("memory_plan"):
            plan = state["memory_plan"]
            return {
                "retrieval_plan": plan,
                "trace": [f"planner: reused memory plan for sources {plan.sources}"],
            }

        # Generate plan via LLM
        try:
            messages = [
                ("system", PLANNER_SYSTEM_PROMPT),
                ("human", f"Query: {query}"),
            ]
            response = self.llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            plan = _parse_plan(raw, query)
        except Exception as e:
            print(f"[Planner] LLM error: {e}. Using default plan.")
            plan = DEFAULT_PLAN.model_copy()
            plan.query_variants = [query]

        return {
            "retrieval_plan": plan,
            "trace": [f"planner: generated plan for sources {plan.sources}"],
        }
