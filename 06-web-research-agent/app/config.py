from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    app_name: str = "Web Research Agent"
    app_version: str = "1.0.0"

    openai_api_key: str = ""
    llm_model: str = "gpt-5-mini"
    serper_api_key: str = ""

    agent_api_key: str = "dev-key-change-me"
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )

    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 60 * 60 * 24 * 7
    max_history_messages: int = 12

    rate_limit_per_minute: int = 10
    monthly_budget_usd: float = 10.0

    search_results_limit: int = 5
    fetch_max_markdown_chars: int = 8000
    max_tool_rounds: int = 4
    crawler_page_timeout_ms: int = 20000

    input_cost_per_1k_tokens_usd: float = 0.00025
    output_cost_per_1k_tokens_usd: float = 0.00200
    search_tool_cost_usd: float = 0.001
    fetch_tool_cost_usd: float = 0.002

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: object) -> list[str] | object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
