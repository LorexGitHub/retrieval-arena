from pydantic import BaseModel
from typing import Optional


class RetrievalResult(BaseModel):
    documents: list[str]
    scores: list[float]
    model_name: str
    top_k: int
    document_ids: list[str] = []


class GenerationResult(BaseModel):
    answer: str
    model_name: str
    retrieval_model: str
    retrieved_documents: list[str]


class EvaluationMetrics(BaseModel):
    # Retrieval metrics (embedding model in isolation)
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision: float = 0.0
    ndcg: float = 0.0
    # Generation metrics (impact on LLM)
    exact_match: bool
    rouge_l_f1: float
    semantic_similarity: float
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    llm_quality_score: Optional[float] = None


class RAGResult(BaseModel):
    query: str
    ground_truth: str
    dataset_used: str
    retrieval: RetrievalResult
    generation: GenerationResult
    evaluation: EvaluationMetrics


class ErrorResult(BaseModel):
    error: str


class ExperimentConfig(BaseModel):
    query: str
    ground_truth: str
    dataset_name: str
    embedding_models: list[str]
    top_k: int = 5


class ExperimentReport(BaseModel):
    query: str
    ground_truth: str
    dataset: str
    results: dict[str, RAGResult | ErrorResult]
    best_model: Optional[str] = None
