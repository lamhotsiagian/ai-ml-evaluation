"""RAG evaluation: indexing, the traced pipeline, and stage-wise metrics."""

from evalcore.rag.index import BM25Index, Chunk, RagIndex, RetrievalHit, sentence_chunk
from evalcore.rag.metrics import (
    ClaimVerifier,
    FaithfulnessReport,
    RetrievalStageReport,
    context_precision,
    evaluate_abstention,
    evaluate_citations,
    evaluate_retrieval_stage,
)
from evalcore.rag.pipeline import LLMReranker, RagPipeline, RagResponse

__all__ = [
    "BM25Index",
    "Chunk",
    "ClaimVerifier",
    "FaithfulnessReport",
    "LLMReranker",
    "RagIndex",
    "RagPipeline",
    "RagResponse",
    "RetrievalHit",
    "RetrievalStageReport",
    "context_precision",
    "evaluate_abstention",
    "evaluate_citations",
    "evaluate_retrieval_stage",
    "sentence_chunk",
]
