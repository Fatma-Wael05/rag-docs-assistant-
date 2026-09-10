"""
Application settings, loaded from environment variables / a .env file.
See .env.example for the full list of configurable values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Vector store / embeddings -- must match what the notebook used in Phase 2
    vector_store_dir: str = "data/vector_store"
    chroma_collection_name: str = "rag_docs"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # LLM
    ollama_model: str = "llama3.1"

    # Retrieval
    retrieval_top_k: int = 4

    # CORS -- comma-separated list of allowed frontend origins
    cors_origins: str = "http://localhost:8501"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
