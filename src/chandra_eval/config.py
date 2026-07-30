"""Typed runtime configuration."""

from pathlib import Path

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from chandra_eval.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Environment-backed settings for local and Tyrone deployments."""

    chandra_base_url: AnyUrl = Field(default="http://127.0.0.1:18083")
    chandra_model: str = Field(default="chandra-ocr-2-q8", min_length=1)
    gemma_base_url: AnyUrl = Field(default="http://127.0.0.1:3001/v1")
    gemma_model: str = Field(default="gemma4-12b", min_length=1)
    gemma_api_key: str = Field(default="no-key", min_length=1)
    llm_request_timeout_seconds: int = Field(default=900, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    gemma_resume_max_tokens: int = Field(default=8192, ge=512)
    chandra_ocr_retry_max_tokens: int = Field(default=8192, ge=512)
    pipeline_output_dir: Path = Path("outputs")
    resume_input_dir: Path = Path("inputs/resumes")
    jd_input_dir: Path = Path("inputs/jds")
    match_output_root: Path = Path("outputs/matches")
    pipeline_debug_raw_responses: bool = False
    allow_partial_evaluation: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("chandra_model", "gemma_model", "gemma_api_key")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


def load_settings() -> Settings:
    """Load settings and raise a project exception with actionable detail."""

    try:
        return Settings()
    except Exception as error:  # pydantic emits detailed validation messages.
        raise ConfigurationError(f"Invalid configuration: {error}") from error
