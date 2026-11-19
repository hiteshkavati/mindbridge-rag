from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_SECRET = "change-me"


class Settings(BaseSettings):
    """All runtime configuration. Environment variables use the MB_ prefix,
    except the provider API keys which keep their usual names."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MB_", extra="ignore", populate_by_name=True
    )

    env: Literal["dev", "prod"] = "dev"
    kb_dir: Path = Path("data/kb")
    helplines_path: Path = Path("data/helplines.json")

    # retrieval
    embedding_backend: Literal["auto", "lsa", "sbert"] = "auto"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    # A question is treated as outside the knowledge base when its IDF-weighted word
    # coverage is below min_coverage (calibrated by `mindbridge eval`). With a sentence
    # embedding model, a dense cosine above min_dense_score can rescue paraphrases.
    min_coverage: float = 0.51
    min_dense_score: float | None = None

    # generation
    llm_provider: Literal["none", "anthropic", "openai", "ollama"] = "none"
    llm_max_tokens: int = 450
    llm_temperature: float = 0.3
    llm_timeout_s: float = 30.0
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = "claude-haiku-5-5"
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # safety
    safety_llm_check: bool = False

    # conversation
    history_turns: int = 6
    session_ttl_s: int = 2 * 60 * 60
    max_message_chars: int = 2000

    # api
    require_auth: bool = True
    jwt_secret: str = _PLACEHOLDER_SECRET
    jwt_ttl_minutes: int = 12 * 60
    db_path: Path = Path("mindbridge.db")
    rate_limit_per_min: int = 20

    @model_validator(mode="after")
    def _check_secret(self) -> Settings:
        if self.env == "prod" and self.require_auth:
            if self.jwt_secret == _PLACEHOLDER_SECRET or len(self.jwt_secret) < 32:
                raise ValueError("MB_JWT_SECRET must be set to 32+ characters when MB_ENV=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
