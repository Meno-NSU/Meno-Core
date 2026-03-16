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

    # Embedding
    local_embedder_path: Optional[Path] = Field(
        default=None,
        validation_alias="LOCAL_EMBEDDER_PATH",
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
    rag_engine_type: Literal["lightrag", "zvec"] = Field(
        default="lightrag",
        validation_alias="RAG_ENGINE_TYPE"
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
    kv_store_text_chunks_path: Path = Field(
        default=Path("resources/lightrag_kg_v3/kv_store_text_chunks.json"),
        validation_alias="KV_STORE_TEXT_CHUNKS_PATH"
    )
    chunk_rag_top_k_dense: int = Field(default=10, validation_alias="CHUNK_RAG_TOP_K_DENSE")
    chunk_rag_top_k_bm25: int = Field(default=10, validation_alias="CHUNK_RAG_TOP_K_BM25")
    chunk_rag_top_k_after_fusion: int = Field(default=15, validation_alias="CHUNK_RAG_TOP_K_AFTER_FUSION")
    chunk_rag_top_n_after_rerank: int = Field(default=5, validation_alias="CHUNK_RAG_TOP_N_AFTER_RERANK")
    chunk_rag_token_budget: int = Field(default=4000, validation_alias="CHUNK_RAG_TOKEN_BUDGET")
    chunk_rag_rewrite_enabled: bool = Field(default=True, validation_alias="CHUNK_RAG_REWRITE_ENABLED")
    chunk_rag_hypothetical_doc_enabled: bool = Field(default=True, validation_alias="CHUNK_RAG_HYPOTHETICAL_DOC_ENABLED")
    chunk_rag_reliability_mode_enabled: bool = Field(default=False, validation_alias="CHUNK_RAG_RELIABILITY_MODE_ENABLED")
    chunk_rag_hallucination_threshold: float = Field(default=0.4, validation_alias="CHUNK_RAG_HALLUCINATION_THRESHOLD")

    # EMBEDDER
    embedder_dim: int = Field(default=768, validation_alias="EMBEDDER_DIM")
    embedder_max_tokens: int = Field(
        default=1024,
        validation_alias="EMBEDDER_MAX_TOKENS",
    )

    # RERANKER
    local_reranker_path: Optional[Path] = Field(
        default=None,
        validation_alias="LOCAL_RERANKER_PATH",
    )

    # LIGHT RAG
    temperature: float = Field(default=0.3, validation_alias="TEMPERATURE")
    query_max_tokens: int = Field(
        default=14_000,
        validation_alias="QUERY_MAX_TOKENS",
    )

    chunk_max_tokens: int = Field(default=1024, validation_alias="CHUNK_MAX_TOKENS")

    entity_max_tokens: int = Field(default=2816, validation_alias="ENTITY_MAX_TOKENS")
    relation_max_tokens: int = Field(default=4096, validation_alias="RELATION_MAX_TOKENS")

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
