from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_base_url: str
    llm_model_name: str

    # Embedding
    local_embedder_path: str

    # RAG
    working_dir: Path
    abbreviations_file: Path

    # LINKS
    urls_path: Path = Path("resources/validated_urls.json")
    max_links: int = 3
    top_k: int = 30
    dist_threshold: float = 0.70

    # EMBEDDER
    embedder_dim: int = 768
    embedder_max_tokens: int = 4096

    # LIGHT RAG
    temperature: float = 0.3
    query_max_tokens: int = 4000

    # LOGS
    log_file_path: Path = Path("/light_rag_log.log")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
        case_sensitive=False,
    )


settings = Settings()
