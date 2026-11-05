"""Векторное хранилище на Qdrant.

По умолчанию локальный режим: база лежит файлом и сервер не нужен.
Для команды задайте QDRANT_URL и QDRANT_API_KEY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Config, load


@dataclass
class Hit:
    score: float
    text: str
    source: str


def client(cfg: Config | None = None) -> QdrantClient:
    cfg = cfg or load()
    if cfg.qdrant_url:
        return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None)
    return QdrantClient(path=cfg.qdrant_path)


def recreate(cli: QdrantClient, cfg: Config, dim: int) -> None:
    """Пересоздать коллекцию. Полная переиндексация."""
    if cli.collection_exists(cfg.collection):
        cli.delete_collection(cfg.collection)
    cli.create_collection(
        cfg.collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert(cli: QdrantClient, cfg: Config, vectors: List[List[float]],
           payloads: List[dict]) -> None:
    points = [
        PointStruct(id=i, vector=vector, payload=payload)
        for i, (vector, payload) in enumerate(zip(vectors, payloads))
    ]
    cli.upsert(cfg.collection, points=points)


def search(cli: QdrantClient, cfg: Config, vector: List[float],
           top_k: int | None = None) -> List[Hit]:
    result = cli.query_points(
        cfg.collection,
        query=vector,
        limit=top_k or cfg.top_k,
        with_payload=True,
    ).points
    return [
        Hit(
            score=point.score,
            text=(point.payload or {}).get("text", ""),
            source=(point.payload or {}).get("source", ""),
        )
        for point in result
    ]
