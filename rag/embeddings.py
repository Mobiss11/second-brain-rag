"""Эмбеддинги через любой OpenAI-совместимый API.

Провайдер задаётся настройками: OpenRouter, Yandex, локальный Ollama или vLLM.
"""
from __future__ import annotations

import time
from typing import Iterable, List

import openai

from .config import Config, load


def _client(cfg: Config) -> openai.OpenAI:
    return openai.OpenAI(api_key=cfg.embed_api_key, base_url=cfg.embed_base_url)


def embed(texts: Iterable[str], cfg: Config | None = None, *, batch: int = 32,
          retries: int = 3) -> List[List[float]]:
    """Вернуть векторы в том же порядке, что и входные тексты."""
    cfg = cfg or load()
    items = list(texts)
    if not items:
        return []
    if not cfg.embed_api_key:
        raise RuntimeError("не задан EMBED_API_KEY (или OPENROUTER_API_KEY)")

    client = _client(cfg)
    out: List[List[float]] = []
    for start in range(0, len(items), batch):
        part = items[start : start + batch]
        for attempt in range(1, retries + 1):
            try:
                response = client.embeddings.create(model=cfg.embed_model, input=part)
                ordered = sorted(response.data, key=lambda item: item.index)
                out.extend([item.embedding for item in ordered])
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(1.5 * attempt)

    for vector in out:
        if len(vector) != cfg.embed_dim:
            raise RuntimeError(
                f"размерность вектора {len(vector)} не совпадает с EMBED_DIM={cfg.embed_dim}. "
                "Проверьте модель и EMBED_DIM, затем переиндексируйте базу."
            )
    return out


def embed_one(text: str, cfg: Config | None = None) -> List[float]:
    return embed([text], cfg)[0]
