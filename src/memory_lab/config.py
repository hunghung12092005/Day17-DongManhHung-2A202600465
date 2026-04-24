from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    memory_lab_mode: str = "offline"
    redis_url: str = "redis://localhost:6379/0"
    token_budget: int = 2200
    max_response_tokens: int = 350
    chroma_dir: Path = DATA_DIR / "chroma"
    episodic_path: Path = DATA_DIR / "episodic_memory.json"
    semantic_seed_path: Path = DATA_DIR / "knowledge_base.json"
    benchmark_path: Path = DATA_DIR / "benchmark_conversations.json"
    report_path: Path = REPORTS_DIR / "benchmark_report.md"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
