"""Prompt templates live in /prompts as plain text so they can be versioned and tuned."""
from __future__ import annotations

import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **kwargs) -> str:
    """Safe templating: only replaces {key} for the given keys (JSON braces are untouched)."""
    text = load_prompt(name)
    for key, value in kwargs.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=1, default=str)
        text = text.replace("{" + key + "}", value)
    return text


def system_prompt() -> str:
    return load_prompt("system_guardrails")
