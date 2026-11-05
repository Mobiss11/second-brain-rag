from pathlib import Path

from rag import query
from rag.config import Config
from rag.store import Hit


def make_config(tmp_path: Path) -> Config:
    return Config(
        embed_base_url="http://localhost/v1",
        embed_api_key="test",
        embed_model="test",
        embed_dim=4,
        llm_base_url="",
        llm_api_key="",
        llm_model="",
        llm_project="",
        qdrant_path=str(tmp_path),
        qdrant_url="",
        qdrant_api_key="",
        collection="kb",
        top_k=2,
        min_score=0.35,
        judge=False,
        knowledge_dir=tmp_path,
    )


def test_escalates_when_score_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(query, "embed_one", lambda text, cfg=None: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(query, "client", lambda cfg=None: object())
    monkeypatch.setattr(
        query, "search",
        lambda cli, cfg, vector: [Hit(score=0.1, text="кусок", source="example.md")],
    )
    text, hits = query.answer("вопрос", cfg=make_config(tmp_path))
    assert text == query.NO_CONTEXT
    assert len(hits) == 1


def test_returns_context_when_model_is_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(query, "embed_one", lambda text, cfg=None: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(query, "client", lambda cfg=None: object())
    monkeypatch.setattr(
        query, "search",
        lambda cli, cfg, vector: [Hit(score=0.9, text="кусок базы", source="example.md")],
    )
    text, hits = query.answer("вопрос", cfg=make_config(tmp_path))
    assert "модель не настроена" in text
    assert "кусок базы" in text
    assert len(hits) == 1
