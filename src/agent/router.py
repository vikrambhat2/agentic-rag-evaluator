from typing import List

from src.models.schemas import AgentState

SOURCE_TO_NODE = {
    "vector_db": "vector_retriever",
    "filesystem": "file_retriever",
    "api": "api_retriever",
}


def route_to_tools(state: AgentState) -> List[str]:
    """
    LangGraph conditional edge function.
    Returns list of next node names based on the retrieval plan's sources.
    """
    plan = state.get("retrieval_plan")
    if not plan:
        return ["vector_retriever"]  # safe default

    next_nodes = []
    for source in plan.sources:
        node = SOURCE_TO_NODE.get(source)
        if node:
            next_nodes.append(node)

    return next_nodes if next_nodes else ["vector_retriever"]
