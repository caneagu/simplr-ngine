from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str
    embedding_dim: int = 1536
    storage_dir: str = "./storage"

    llm_provider: str = "openai"  # openai | openrouter
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    app_base_url: str = "http://localhost:8000"
    magiclink_ttl_minutes: int = 15
    session_days: int = 7
    cookie_secure: bool = False

    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_sender: Optional[str] = None
    smtp_sender_name: Optional[str] = None
    email_logo_url: Optional[str] = None
    email_brand_name: str = "Simplr"


settings = Settings()
