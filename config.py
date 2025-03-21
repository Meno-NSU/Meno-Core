from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    openai_api_key: str
    openai_base_url: str
    llm_model_name: str

    class Config:
        env_file = ".env"


settings = Settings()
