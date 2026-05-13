from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen3.5:9b"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    DART_API_KEY: str = ""
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""

    class Config:
        env_file = ".env"

settings = Settings()