from pathlib import Path

from rag.chunking import load_knowledge, split_markdown


def test_split_keeps_blocks_and_drops_short():
    text = "## Первый вопрос\n\nОтвет на первый вопрос, достаточно длинный.\n\n---\n\n## Второй\n\nКороткий.\n\n---\n\n## Третий вопрос\n\nОтвет на третий вопрос, тоже длинный."
    chunks = split_markdown(text, "example.md")
    assert len(chunks) == 2
    assert chunks[0]["title"] == "Первый вопрос"
    assert chunks[1]["title"] == "Третий вопрос"
    assert all(chunk["source"] == "example.md" for chunk in chunks)


def test_service_lines_are_removed():
    text = "> служебная строка\n\n## Вопрос\n\nОтвет на вопрос, который длиннее сорока символов."
    chunks = split_markdown(text, "example.md")
    assert len(chunks) == 1
    assert "служебная" not in chunks[0]["text"]


def test_load_knowledge_reads_markdown(tmp_path: Path):
    (tmp_path / "a.md").write_text("## Вопрос\n\nОтвет достаточной длины для индексации.", encoding="utf-8")
    (tmp_path / "b.md").write_text("## Второй\n\nЕщё один ответ достаточной длины для индексации.", encoding="utf-8")
    chunks = load_knowledge(tmp_path)
    assert len(chunks) == 2
    assert {chunk["source"] for chunk in chunks} == {"a.md", "b.md"}
