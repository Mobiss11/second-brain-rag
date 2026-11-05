"""Проверка черновика ответа перед выдачей.

Второй проход модели: все ли цифры есть в источнике, нет ли обещаний.
Лучший результат - когда проверяет другая модель, но для старта хватит той же.
Отключается переменной JUDGE=0.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import openai

from .config import Config, load

LOG = Path(__file__).resolve().parent.parent / "judge_log.jsonl"

SYSTEM = """Ты строгий проверяющий. Тебе дан ИСТОЧНИК и ЧЕРНОВИК ответа.

Проверь:
1. Все цифры, даты и названия есть в источнике.
2. Нет обещаний, которых нет в источнике.
3. Нет ссылок на то, чего в источнике нет.

Верни строго JSON:
{"ok": true|false, "issues": ["..."], "final": "<исправленный текст или пусто>"}

Если черновик в порядке - ok=true, issues=[], final пустой.
Если есть ошибки - ok=false, перечисли их и верни исправленный текст в final,
не добавляя ничего сверх источника."""


def _log(issues: list, changed: bool) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps({"issues": issues, "changed": changed}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def review(draft: str, source: str, cfg: Config | None = None) -> str:
    """Вернуть проверенный текст. При любой ошибке проверки - исходный черновик."""
    cfg = cfg or load()
    if not cfg.llm_api_key or not cfg.llm_model or not cfg.llm_base_url or not draft.strip():
        return draft
    try:
        kwargs = {"api_key": cfg.llm_api_key, "base_url": cfg.llm_base_url}
        if cfg.llm_project:
            kwargs["project"] = cfg.llm_project
        response = openai.OpenAI(**kwargs).chat.completions.create(
            model=cfg.llm_model,
            temperature=0.1,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"ИСТОЧНИК:\n{source}\n\nЧЕРНОВИК:\n{draft}"},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return draft
        data = json.loads(raw[start : end + 1])
        issues = data.get("issues") or []
        final = (data.get("final") or "").strip()
        if issues:
            _log(issues, bool(final))
        if data.get("ok") is False and final:
            return final
        return draft
    except Exception:
        return draft
