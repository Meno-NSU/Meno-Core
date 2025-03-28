from pydantic_settings import BaseSettings
from pathlib import Path


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

    class Config:
        env_file = ".env"


settings = Settings()
