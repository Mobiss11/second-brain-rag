"""Настройки из окружения. Ключи в код не попадают."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    embed_dim: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_project: str
    qdrant_path: str
    qdrant_url: str
    qdrant_api_key: str
    collection: str
    top_k: int
    min_score: float
    judge: bool
    knowledge_dir: Path


def load() -> Config:
    return Config(
        embed_base_url=_str("EMBED_BASE_URL", "https://openrouter.ai/api/v1"),
        embed_api_key=_str("EMBED_API_KEY") or _str("OPENROUTER_API_KEY"),
        embed_model=_str("EMBED_MODEL", "perplexity/pplx-embed-v1-4b"),
        embed_dim=_int("EMBED_DIM", 2560),
        llm_base_url=_str("LLM_BASE_URL"),
        llm_api_key=_str("LLM_API_KEY"),
        llm_model=_str("LLM_MODEL"),
        llm_project=_str("LLM_PROJECT"),
        qdrant_path=_str("QDRANT_PATH", str(ROOT / "qdrant_data")),
        qdrant_url=_str("QDRANT_URL"),
        qdrant_api_key=_str("QDRANT_API_KEY"),
        collection=_str("COLLECTION", "kb"),
        top_k=_int("TOP_K", 4),
        min_score=_float("MIN_SCORE", 0.35),
        judge=_bool("JUDGE", True),
        knowledge_dir=Path(_str("KNOWLEDGE_DIR", str(ROOT / "knowledge"))),
    )
