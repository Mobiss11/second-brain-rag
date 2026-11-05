"""Чтение базы знаний и нарезка на блоки.

Формат: markdown, один блок = один самодостаточный факт или вопрос-ответ.
Блоки разделены строкой из трёх дефисов.
"""
from __future__ import annotations

import re
from pathlib import Path


def split_markdown(text: str, source: str, min_len: int = 40) -> list[dict]:
    """Разрезать файл на блоки. Слишком короткие блоки отбрасываются."""
    chunks: list[dict] = []
    for block in re.split(r"\n---\n", text):
        lines = [
            line
            for line in block.strip().splitlines()
            if not line.strip().startswith(">")  # служебные строки-цитаты
        ]
        body = "\n".join(lines).strip()
        if len(body) < min_len:
            continue
        title = body.splitlines()[0].lstrip("# ").strip()[:120]
        chunks.append({"text": body, "source": source, "title": title})
    return chunks


def load_knowledge(directory: Path, min_len: int = 40) -> list[dict]:
    """Прочитать все .md из папки и вернуть блоки с источником."""
    chunks: list[dict] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(split_markdown(path.read_text(encoding="utf-8"), path.name, min_len))
    return chunks
