from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "NSU_OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    llm_model_name: str = Field(
        default=None,
        validation_alias="LLM_MODEL_NAME",
    )

    # Embedding
    local_embedder_path: Path = Field(
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
        validation_alias="ABBREVIATIONS_PATH",
    )
    query_mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="naive",
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
    top_k: int = Field(default=15, validation_alias="TOP_K")
    dist_threshold: float = Field(default=0.70, validation_alias="DIST_THRESHOLD")
    correct_dist_threshold: float = Field(
        default=0.1,
        validation_alias="CORRECT_DIST_THRESHOLD",
    )

    # EMBEDDER
    embedder_dim: int = Field(default=768, validation_alias="EMBEDDER_DIM")
    embedder_max_tokens: int = Field(
        default=4096,
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
        default=4000,
        validation_alias="QUERY_MAX_TOKENS",
    )

    # LOGS
    log_file_path: Path = Field(
        default=Path("logs/light_rag_log.log"),
        validation_alias="LOG_FILE_PATH",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=True,
        populate_by_name=True,
    )


settings = Settings()
