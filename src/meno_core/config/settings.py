from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "NSU_OPENAI_API_KEY"),
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    llm_model_name: Optional[str] = Field(
        default=None,
        validation_alias="LLM_MODEL_NAME",
    )

    # Embeddings / Reranker
    multilingual_embedder_path: str = Field(
        default="Alibaba-NLP/gte-multilingual-base",
        validation_alias="MULTILINGUAL_EMBEDDER_PATH",
    )
    rus_embedder_path: str = Field(
        default="deepvk/USER2-base",
        validation_alias="RUS_EMBEDDER_PATH",
    )
    reranker_path: str = Field(
        default="Qwen/Qwen3-Reranker-4B",
        validation_alias="RERANKER_PATH",
    )

    # RAG
    working_dir: Optional[Path] = Field(
        default=None,
        validation_alias="WORKING_DIR",
    )
    abbreviations_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("ABBREVIATIONS_PATH", "ABBREVIATIONS_FILE"),
    )
    query_mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="mix",
        validation_alias="QUERY_MODE",
    )

    # LINKS
    enable_links_addition: bool = Field(
        default=True,
        validation_alias="ENABLE_LINKS_ADDITION",
    )
    enable_links_correction: bool = Field(
        default=True,
        validation_alias="ENABLE_LINKS_CORRECTION",
    )
    urls_path: Path = Field(
        default=Path("../../../resources/validated_urls.json"),
        validation_alias="URLS_PATH",
    )
    max_links: int = Field(default=5, validation_alias="MAX_LINKS")
    top_k: int = Field(default=100, validation_alias="TOP_K")
    chunk_top_k: int = Field(default=12, validation_alias="CHUNK_TOP_K")
    dist_threshold: float = Field(default=0.70, validation_alias="DIST_THRESHOLD")
    correct_dist_threshold: float = Field(
        default=0.1,
        validation_alias="CORRECT_DIST_THRESHOLD",
    )

    # CHUNK RAG
    chunk_rag_corpus_path: Path = Field(
        default=Path("resources/data/chunk_rag_corpus_512.jsonl"),
        validation_alias="CHUNK_RAG_CORPUS_PATH"
    )
    chunk_rag_data_path: Path = Field(
        default=Path("resources/chunk_rag_data"),
        validation_alias="CHUNK_RAG_DATA_PATH"
    )
    chunk_rag_auto_rebuild: bool = Field(
        default=False,
        validation_alias="CHUNK_RAG_AUTO_REBUILD",
    )
    chunk_rag_top_k_dense_multilingual: int = Field(
        default=10,
        validation_alias="CHUNK_RAG_TOP_K_DENSE_MULTILINGUAL",
    )
    chunk_rag_top_k_dense_russian: int = Field(
        default=10,
        validation_alias="CHUNK_RAG_TOP_K_DENSE_RUSSIAN",
    )
    chunk_rag_top_k_bm25: int = Field(default=10, validation_alias="CHUNK_RAG_TOP_K_BM25")
    chunk_rag_top_k_after_fusion: int = Field(default=15, validation_alias="CHUNK_RAG_TOP_K_AFTER_FUSION")
    chunk_rag_top_n_after_rerank: int = Field(default=5, validation_alias="CHUNK_RAG_TOP_N_AFTER_RERANK")
    chunk_rag_token_budget: int = Field(default=4000, validation_alias="CHUNK_RAG_TOKEN_BUDGET")
    chunk_rag_rewrite_enabled: bool = Field(default=True, validation_alias="CHUNK_RAG_REWRITE_ENABLED")
    chunk_rag_hypothetical_doc_enabled: bool = Field(default=True, validation_alias="CHUNK_RAG_HYPOTHETICAL_DOC_ENABLED")
    chunk_rag_reliability_mode_enabled: bool = Field(default=False, validation_alias="CHUNK_RAG_RELIABILITY_MODE_ENABLED")
    chunk_rag_hallucination_threshold: float = Field(default=0.4, validation_alias="CHUNK_RAG_HALLUCINATION_THRESHOLD")
    chunk_rag_reranker_filter_threshold: float = Field(default=0.0, validation_alias="CHUNK_RAG_RERANKER_FILTER_THRESHOLD")
    chunk_rag_fusion_weight_multilingual: float = Field(
        default=0.35,
        validation_alias="CHUNK_RAG_FUSION_WEIGHT_MULTILINGUAL",
    )
    chunk_rag_fusion_weight_russian: float = Field(
        default=0.35,
        validation_alias="CHUNK_RAG_FUSION_WEIGHT_RUSSIAN",
    )
    chunk_rag_fusion_weight_bm25: float = Field(
        default=0.30,
        validation_alias="CHUNK_RAG_FUSION_WEIGHT_BM25",
    )
    chunk_rag_debug_retrieval: bool = Field(
        default=False,
        validation_alias="CHUNK_RAG_DEBUG_RETRIEVAL",
    )
    chunk_rag_retrieval_log_level: str = Field(
        default="INFO",
        validation_alias="CHUNK_RAG_RETRIEVAL_LOG_LEVEL",
    )
    chunk_rag_retrieval_preview_k: int = Field(
        default=5,
        validation_alias="CHUNK_RAG_RETRIEVAL_PREVIEW_K",
    )

    # EMBEDDER
    embedder_dim: int = Field(default=768, validation_alias="EMBEDDER_DIM")
    embedder_max_tokens: int = Field(
        default=1024,
        validation_alias="EMBEDDER_MAX_TOKENS",
    )

    # LIGHT RAG
    temperature: float = Field(default=0.3, validation_alias="TEMPERATURE")
    query_max_tokens: int = Field(
        default=14_000,
        validation_alias="QUERY_MAX_TOKENS",
    )

    chunk_max_tokens: int = Field(default=1024, validation_alias="CHUNK_MAX_TOKENS")

    entity_max_tokens: int = Field(default=2000, validation_alias="ENTITY_MAX_TOKENS")
    relation_max_tokens: int = Field(default=3000, validation_alias="RELATION_MAX_TOKENS")

    # LOGS
    log_file_path: Path = Field(
        default=Path("logs/light_rag_log.log"),
        validation_alias="LOG_FILE_PATH",
    )

    vllm_endpoints: str = Field(
        default="",
        validation_alias="VLLM_ENDPOINTS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=True,
        populate_by_name=True,
    )


settings = Settings()
