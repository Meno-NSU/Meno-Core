from pydantic import BaseModel, Field

from meno_core.config.settings import settings


class ChunkRagConfig(BaseModel):
    """
    Configuration for the Chunk RAG Module, mapping to global settings defaults.
    """
    top_k_dense_multilingual: int = Field(default_factory=lambda: settings.chunk_rag_top_k_dense_multilingual)
    top_k_dense_russian: int = Field(default_factory=lambda: settings.chunk_rag_top_k_dense_russian)
    top_k_bm25: int = Field(default_factory=lambda: settings.chunk_rag_top_k_bm25)
    top_k_after_fusion: int = Field(default_factory=lambda: settings.chunk_rag_top_k_after_fusion)
    top_n_after_rerank: int = Field(default_factory=lambda: settings.chunk_rag_top_n_after_rerank)
    token_budget: int = Field(default_factory=lambda: settings.chunk_rag_token_budget)

    # Boolean flags for enabling specific components
    rewrite_enabled: bool = Field(default_factory=lambda: settings.chunk_rag_rewrite_enabled)
    hypothetical_doc_enabled: bool = Field(default_factory=lambda: settings.chunk_rag_hypothetical_doc_enabled)
    reliability_mode_enabled: bool = Field(default_factory=lambda: settings.chunk_rag_reliability_mode_enabled)
    debug_retrieval: bool = Field(default_factory=lambda: settings.chunk_rag_debug_retrieval)
    retrieval_log_level: str = Field(default_factory=lambda: settings.chunk_rag_retrieval_log_level)
    retrieval_preview_k: int = Field(default_factory=lambda: settings.chunk_rag_retrieval_preview_k)

    # Thresholds
    hallucination_threshold: float = Field(default_factory=lambda: settings.chunk_rag_hallucination_threshold)
    reranker_filter_threshold: float = Field(default_factory=lambda: settings.chunk_rag_reranker_filter_threshold)
    fusion_weight_multilingual: float = Field(default_factory=lambda: settings.chunk_rag_fusion_weight_multilingual)
    fusion_weight_russian: float = Field(default_factory=lambda: settings.chunk_rag_fusion_weight_russian)
    fusion_weight_bm25: float = Field(default_factory=lambda: settings.chunk_rag_fusion_weight_bm25)
