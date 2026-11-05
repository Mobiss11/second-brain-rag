"""Индексация базы знаний.

    python -m rag.ingest --dry-run   показать нарезку, без сети и записи
    python -m rag.ingest             построить индекс
"""
from __future__ import annotations

import argparse

from . import store
from .chunking import load_knowledge
from .config import load
from .embeddings import embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация базы знаний в Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="только показать блоки")
    args = parser.parse_args()

    cfg = load()
    chunks = load_knowledge(cfg.knowledge_dir)
    if not chunks:
        print(f"в {cfg.knowledge_dir} нет блоков. Проверьте формат: блоки через ---")
        return

    print(f"блоков: {len(chunks)}")
    if args.dry_run:
        for chunk in chunks[:5]:
            print(f"\n[{chunk['source']}] {chunk['title']}")
            print(chunk["text"][:200].replace("\n", " "))
        if len(chunks) > 5:
            print(f"\n... ещё {len(chunks) - 5}")
        return

    vectors = embed([chunk["text"] for chunk in chunks], cfg)
    print(f"векторов: {len(vectors)} x {len(vectors[0])}")

    cli = store.client(cfg)
    store.recreate(cli, cfg, len(vectors[0]))
    store.upsert(cli, cfg, vectors, chunks)
    print(f"готово: коллекция {cfg.collection} в {cfg.qdrant_url or cfg.qdrant_path}")


if __name__ == "__main__":
    main()
