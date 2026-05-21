from functools import lru_cache

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    # LangChain
    langchain_api_key: str = Field(alias="LANGCHAIN_API_KEY")
    langchain_tracing: str = Field(alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(alias="LANGCHAIN_ENDPOINT")
    langchain_project: str = Field(alias="LANGCHAIN_PROJECT")

    # Models
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    hf_token: str = Field(alias="HF_TOKEN")
    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY")

    # Postgres (optional; chat memory disabled when unset)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Qdrant
    qdrant_api_key: str = Field(alias="QDRANT_API_KEY")
    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_collection_name: str = Field(alias="QDRANT_COLLECTION_NAME")

    # Azure Speech
    azure_speech_key: str = Field(alias="speech_key")
    azure_region: str = Field(alias="region")

    # Algolia
    algolia_index_name: str = Field(alias="INDEX_NAME")
    algolia_application_id: str = Field(alias="APPLICATION_ID")
    algolia_search_api_key: str = Field(alias="SEARCH_API_KEY")
    algolia_write_api_key: str = Field(alias="WRITE_API_KEY")
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")

    # Langfuse (optional — tracing disabled when unset)
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

@lru_cache
def get_settings() -> Settings:
    return Settings()


my_settings = get_settings()