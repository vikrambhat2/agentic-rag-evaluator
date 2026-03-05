"""
LangGraph RAG graph: memory_check → planner → [retrievers] → evaluator → generator → memory_update
"""

from typing import List

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.agent.evaluator import EvaluatorNode, needs_refinement
from src.agent.memory import MemoryStore
from src.agent.planner import PlannerNode
from src.agent.router import route_to_tools
from src.models.schemas import AgentState, RetrievalResult
from src.tools.api_tool import APITool
from src.tools.file_tool import FileTool
from src.tools.vector_tool import VectorTool

# ─────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────

memory_store = MemoryStore()
planner_node = PlannerNode()
evaluator_node = EvaluatorNode()

vector_tool = VectorTool()
file_tool = FileTool()
api_tool = APITool()

llm = ChatOllama(model="llama3.2", temperature=0.1)


def memory_check_node(state: AgentState) -> AgentState:
    """Check memory store for a cached plan for this query."""
    query = state["query"]
    cached_plan = memory_store.check(query)
    if cached_plan:
        state["memory_hit"] = True
        state["memory_plan"] = cached_plan
        state["trace"].append(f"memory_check: HIT — cached plan for sources {cached_plan.sources}")
    else:
        state["memory_hit"] = False
        state["memory_plan"] = None
        state["trace"].append("memory_check: MISS — no cached plan found")
    return state


def planner_fn(state: AgentState) -> AgentState:
    return planner_node(state)


def vector_retriever_node(state: AgentState) -> AgentState:
    """Retrieve chunks from ChromaDB."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)
    results: List[RetrievalResult] = list(state.get("retrieval_results", []))

    if plan and plan.query_variants:
        # Use variant for vector_db if available
        idx = plan.sources.index("vector_db") if "vector_db" in plan.sources else 0
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
        relevance_score=0.0,  # will be scored by evaluator
        iteration=iteration,
    )
    results.append(result)
    state["retrieval_results"] = results
    state["trace"].append(
        f"vector_retriever: retrieved {len(chunks)} chunks (iteration {iteration})"
    )
    return state


def file_retriever_node(state: AgentState) -> AgentState:
    """Retrieve from internal notes files."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)
    results: List[RetrievalResult] = list(state.get("retrieval_results", []))

    if plan and plan.query_variants:
        idx = plan.sources.index("filesystem") if "filesystem" in plan.sources else 0
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
    results.append(result)
    state["retrieval_results"] = results
    state["trace"].append(
        f"file_retriever: retrieved {len(chunks)} file(s) (iteration {iteration})"
    )
    return state


def api_retriever_node(state: AgentState) -> AgentState:
    """Retrieve from mock API."""
    plan = state.get("retrieval_plan")
    iteration = state.get("iteration", 0)
    results: List[RetrievalResult] = list(state.get("retrieval_results", []))

    if plan and plan.query_variants:
        idx = plan.sources.index("api") if "api" in plan.sources else 0
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
    results.append(result)
    state["retrieval_results"] = results
    state["trace"].append(
        f"api_retriever: retrieved {len(chunks)} entries (iteration {iteration})"
    )
    return state


def evaluator_fn(state: AgentState) -> AgentState:
    return evaluator_node(state)


def generator_node(state: AgentState) -> AgentState:
    """Generate the final answer using all retrieved chunks as context."""
    query = state["query"]
    results = state.get("retrieval_results", [])

    # Collect all chunks across all sources and iterations
    all_chunks = []
    for r in results:
        all_chunks.extend(r.chunks)

    if all_chunks:
        context = "\n\n---\n\n".join(all_chunks[:10])  # cap at 10 chunks
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

    state["final_answer"] = answer
    state["trace"].append(
        f"generator: answer generated using {len(all_chunks)} total chunks"
    )
    return state


def memory_update_node(state: AgentState) -> AgentState:
    """Save the successful plan to memory if quality is sufficient."""
    plan = state.get("retrieval_plan")
    results = state.get("retrieval_results", [])

    if not plan or not results:
        state["trace"].append("memory_update: skipped — no plan or results")
        return state

    scored = [r for r in results if r.relevance_score > 0]
    if scored:
        avg_relevance = sum(r.relevance_score for r in scored) / len(scored)
    else:
        avg_relevance = 0.0

    memory_store.save(state["query"], plan, avg_relevance)

    if avg_relevance >= 0.75:
        state["trace"].append(
            f"memory_update: plan saved (avg_relevance={avg_relevance:.2f})"
        )
    else:
        state["trace"].append(
            f"memory_update: plan NOT saved (avg_relevance={avg_relevance:.2f} < 0.75)"
        )

    return state


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("memory_check", memory_check_node)
    graph.add_node("planner", planner_fn)
    graph.add_node("vector_retriever", vector_retriever_node)
    graph.add_node("file_retriever", file_retriever_node)
    graph.add_node("api_retriever", api_retriever_node)
    graph.add_node("evaluator", evaluator_fn)
    graph.add_node("generator", generator_node)
    graph.add_node("memory_update", memory_update_node)

    # Linear edges
    graph.add_edge(START, "memory_check")
    graph.add_edge("memory_check", "planner")

    # Conditional fan-out from planner to retriever nodes
    graph.add_conditional_edges(
        "planner",
        route_to_tools,
        {
            "vector_retriever": "vector_retriever",
            "file_retriever": "file_retriever",
            "api_retriever": "api_retriever",
        },
    )

    # All retrievers converge to evaluator
    graph.add_edge("vector_retriever", "evaluator")
    graph.add_edge("file_retriever", "evaluator")
    graph.add_edge("api_retriever", "evaluator")

    # Conditional edge from evaluator: refine or generate
    graph.add_conditional_edges(
        "evaluator",
        needs_refinement,
        {
            "planner": "planner",
            "generator": "generator",
        },
    )

    # Final edges
    graph.add_edge("generator", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile()


# Compile the graph once at import time
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
    final_state = rag_graph.invoke(initial_state)
    return final_state
