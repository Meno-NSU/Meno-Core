from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    llm_model_name: Optional[str] = None

    # Embedding
    local_embedder_path: Optional[str] = None

    # RAG
    working_dir: Optional[Path] = None
    abbreviations_file: Optional[Path] = None
    query_mode: Literal["local", "global", "hybrid", "naive", "mix"] = "naive"

    # LINKS
    enable_links_addition: bool = True
    enable_links_correction: bool = True
    urls_path: Path = Path("../../../resources/validated_urls.json")
    max_links: int = 5
    top_k: int = 15
    dist_threshold: float = 0.70
    correct_dist_threshold: float = 0.1

    # EMBEDDER
    embedder_dim: int = 768
    embedder_max_tokens: int = 4096

    # RERANKER
    local_reranker_path: Optional[Path] = None

    # LIGHT RAG
    temperature: float = 0.3
    query_max_tokens: int = 4000

    # LOGS
    log_file_path: Path = Path("logs/light_rag_log.log")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=False,
    )


settings: Settings = Settings()
