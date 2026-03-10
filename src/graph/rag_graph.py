"""
LangGraph RAG graph: memory_check → planner → [retrievers] → evaluator → generator → memory_update

All node functions return partial dicts (not mutated full state) so that LangGraph's
Annotated reducers for `retrieval_results` and `trace` work correctly during parallel fan-out.
"""

from typing import Dict, List, Any

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.agent.evaluator import EvaluatorNode, needs_refinement
from src.agent.memory import MemoryStore
from src.agent.planner import PlannerNode
from src.agent.router import route_to_tools
from src.models.schemas import AgentState, RetrievalPlan, RetrievalResult
from src.tools.api_tool import APITool
from src.tools.file_tool import FileTool
from src.tools.vector_tool import VectorTool

# ─────────────────────────────────────────────
# Singletons (initialised once at import)
# ─────────────────────────────────────────────

memory_store = MemoryStore()
planner_node = PlannerNode()
evaluator_node = EvaluatorNode()

vector_tool = VectorTool()
file_tool = FileTool()
api_tool = APITool()

llm = ChatOllama(model="llama3.2", temperature=0.1)


# ─────────────────────────────────────────────
# Node functions — all return partial dicts
# ─────────────────────────────────────────────

def memory_check_node(state: AgentState) -> Dict[str, Any]:
    """Check memory store for a cached plan for this query."""
    query = state["query"]
    cached_plan = memory_store.check(query)
    if cached_plan:
        return {
            "memory_hit": True,
            "memory_plan": cached_plan,
            "trace": [f"memory_check: HIT — cached plan for sources {cached_plan.sources}"],
        }
    return {
        "memory_hit": False,
        "memory_plan": None,
        "trace": ["memory_check: MISS — no cached plan found"],
    }


def planner_fn(state: AgentState) -> Dict[str, Any]:
    """Delegate to PlannerNode; returns partial dict."""
    return planner_node(state)


def vector_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve chunks from ChromaDB. Returns only new retrieval result."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)

    if plan and plan.query_variants and "vector_db" in plan.sources:
        idx = plan.sources.index("vector_db")
        query = plan.query_variants[idx] if idx < len(plan.query_variants) else state["query"]
    else:
        query = state["query"]

    try:
        chunks = vector_tool.retrieve(query, k=3)
    except Exception as e:
        print(f"[vector_retriever] Error: {e}")
        chunks = []

    result = RetrievalResult(
        source="vector_db",
        chunks=chunks,
        relevance_score=0.0,
        iteration=iteration,
    )
    return {
        "retrieval_results": [result],
        "trace": [f"vector_retriever: retrieved {len(chunks)} chunks (iteration {iteration})"],
    }


def file_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve from internal notes. Returns only new retrieval result."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)

    if plan and plan.query_variants and "filesystem" in plan.sources:
        idx = plan.sources.index("filesystem")
        query = plan.query_variants[idx] if idx < len(plan.query_variants) else state["query"]
    else:
        query = state["query"]

    try:
        chunks = file_tool.retrieve(query, top_k=3)
    except Exception as e:
        print(f"[file_retriever] Error: {e}")
        chunks = []

    result = RetrievalResult(
        source="filesystem",
        chunks=chunks,
        relevance_score=0.0,
        iteration=iteration,
    )
    return {
        "retrieval_results": [result],
        "trace": [f"file_retriever: retrieved {len(chunks)} file(s) (iteration {iteration})"],
    }


def api_retriever_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve from mock API. Returns only new retrieval result."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)

    if plan and plan.query_variants and "api" in plan.sources:
        idx = plan.sources.index("api")
        query = plan.query_variants[idx] if idx < len(plan.query_variants) else state["query"]
    else:
        query = state["query"]

    try:
        chunks = api_tool.retrieve(query)
    except Exception as e:
        print(f"[api_retriever] Error: {e}")
        chunks = []

    result = RetrievalResult(
        source="api",
        chunks=chunks,
        relevance_score=0.0,
        iteration=iteration,
    )
    return {
        "retrieval_results": [result],
        "trace": [f"api_retriever: retrieved {len(chunks)} entries (iteration {iteration})"],
    }


def evaluator_fn(state: AgentState) -> Dict[str, Any]:
    """Delegate to EvaluatorNode; returns partial dict."""
    return evaluator_node(state)


def generator_node(state: AgentState) -> Dict[str, Any]:
    """Generate the final answer using all retrieved chunks as context."""
    query = state["query"]
    results = state.get("retrieval_results", [])

    all_chunks = []
    for r in results:
        all_chunks.extend(r.chunks)

    if all_chunks:
        context = "\n\n---\n\n".join(all_chunks[:10])
    else:
        context = "No relevant information was retrieved."

    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain enough information, "
        "say so clearly. Be concise and accurate."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[generator] LLM error: {e}")
        answer = f"Unable to generate answer due to an error: {e}"

    return {
        "final_answer": answer,
        "trace": [f"generator: answer generated using {len(all_chunks)} total chunks"],
    }


def memory_update_node(state: AgentState) -> Dict[str, Any]:
    """Save the successful plan to memory if quality is sufficient."""
    plan = state.get("retrieval_plan")
    results = state.get("retrieval_results", [])

    if not plan or not results:
        return {"trace": ["memory_update: skipped — no plan or results"]}

    scored = [r for r in results if r.relevance_score > 0]
    avg_relevance = sum(r.relevance_score for r in scored) / len(scored) if scored else 0.0

    memory_store.save(state["query"], plan, avg_relevance)

    if avg_relevance >= 0.75:
        msg = f"memory_update: plan saved (avg_relevance={avg_relevance:.2f})"
    else:
        msg = f"memory_update: plan NOT saved (avg_relevance={avg_relevance:.2f} < 0.75)"

    return {"trace": [msg]}


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("memory_check", memory_check_node)
    graph.add_node("planner", planner_fn)
    graph.add_node("vector_retriever", vector_retriever_node)
    graph.add_node("file_retriever", file_retriever_node)
    graph.add_node("api_retriever", api_retriever_node)
    graph.add_node("evaluator", evaluator_fn)
    graph.add_node("generator", generator_node)
    graph.add_node("memory_update", memory_update_node)

    graph.add_edge(START, "memory_check")
    graph.add_edge("memory_check", "planner")

    # Conditional fan-out: planner → 1–3 retriever nodes (run in parallel when multiple selected)
    graph.add_conditional_edges(
        "planner",
        route_to_tools,
        {
            "vector_retriever": "vector_retriever",
            "file_retriever": "file_retriever",
            "api_retriever": "api_retriever",
        },
    )

    # All active retrievers converge to evaluator
    graph.add_edge("vector_retriever", "evaluator")
    graph.add_edge("file_retriever", "evaluator")
    graph.add_edge("api_retriever", "evaluator")

    # Evaluator decides: refine (→ planner) or proceed (→ generator)
    graph.add_conditional_edges(
        "evaluator",
        needs_refinement,
        {
            "planner": "planner",
            "generator": "generator",
        },
    )

    graph.add_edge("generator", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile()


# Compile once at import time
rag_graph = build_graph()


def run_query(query: str) -> AgentState:
    """Run the full RAG graph for a single query and return the final state."""
    initial_state: AgentState = {
        "query": query,
        "retrieval_plan": None,
        "retrieval_results": [],
        "final_answer": "",
        "iteration": 0,
        "memory_hit": False,
        "memory_plan": None,
        "trace": [],
    }
    return rag_graph.invoke(initial_state)
