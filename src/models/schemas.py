from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
from typing_extensions import TypedDict

SourceType = Literal["vector_db", "filesystem", "api"]


class RetrievalPlan(BaseModel):
    sources: List[SourceType]
    strategy: str               # "semantic", "keyword", "structured"
    query_variants: List[str]   # rewritten query forms for each source
    confidence_threshold: float
    max_iterations: int


class RetrievalResult(BaseModel):
    source: SourceType
    chunks: List[str]
    relevance_score: float      # scored by evaluator node
    iteration: int


class AgentState(TypedDict):
    query: str
    retrieval_plan: Optional[RetrievalPlan]
    retrieval_results: List[RetrievalResult]
    final_answer: str
    iteration: int
    memory_hit: bool
    memory_plan: Optional[RetrievalPlan]
    trace: List[str]            # log of node decisions for evaluation


class TestCase(BaseModel):
    query: str
    ground_truth: str
    expected_sources: List[SourceType]


class LayerScore(BaseModel):
    layer: int
    name: str
    score: float
    details: str


class AgenticEvalResult(BaseModel):
    query: str
    expected_sources: List[SourceType]
    planned_sources: List[SourceType]
    final_answer: str
    retrieval_results: List[RetrievalResult]
    memory_hit: bool
    trace: List[str]
    layer_scores: List[LayerScore]
    overall_score: float        # weighted average of all 5 layers


class AgenticEvalReport(BaseModel):
    results: List[AgenticEvalResult]
    avg_layer1_plan: float
    avg_layer2_retrieval: float
    avg_layer3_refinement: float
    avg_layer4_memory: float
    avg_layer5_alignment: float
    avg_overall: float
