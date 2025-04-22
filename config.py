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
    urls_path: Path
    max_links: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
    )


settings = Settings()
