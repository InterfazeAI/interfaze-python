from __future__ import annotations

from typing import Any, List, Literal, Optional

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

TaskName = Literal[
    "ocr",
    "object_detection",
    "gui_detection",
    "web_search",
    "scraper",
    "translate",
    "speech_to_text",
    "forecast",
]

GuardCode = Literal[
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S1_IMAGE",
    "S12_IMAGE",
    "S15_IMAGE",
    "ALL",
]

# Interfaze accepts these reasoning levels (wider than the OpenAI enum).
ReasoningEffort = Literal["minimal", "low", "medium", "high", "on", "off", "auto"]


class Precontext(BaseModel):
    """One internal task's raw output, surfaced in ``response.precontext``."""

    name: str
    result: Any = None


class InterfazeChatCompletion(ChatCompletion):
    """A chat completion extended with the fields Interfaze adds.

    openai-python already preserves these as pydantic extras (``extra='allow'``); this
    subclass merely gives them declared, typed attributes for IDE/type-checker support.
    """

    precontext: Optional[List[Precontext]] = None
    """Present when internal tools ran (OCR / web search / scrape / STT / forecast / …)."""
    reasoning: Optional[str] = None
    """Reasoning text — present with ``reasoning_effort='high'`` and no schema."""
    vcache: bool = False
    """Whether the semantic cache was hit."""
    debug: Optional[Any] = None
    """Admin-only debug payload (requires ``admin_key``)."""
