from __future__ import annotations

from typing import Any, Generic, List, Literal, Optional, TypeVar

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel, ConfigDict

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

ReasoningEffort = Literal["minimal", "low", "medium", "high", "on", "off", "auto"]


class Precontext(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: Optional[str] = None
    result: Any = None


class InterfazeChatCompletion(ChatCompletion):
    precontext: Optional[List[Precontext]] = None
    reasoning: Optional[str] = None
    vcache: bool = False
    debug: Optional[Any] = None


ContentType = TypeVar("ContentType")


class InterfazeParsedChatCompletion(ParsedChatCompletion[ContentType], Generic[ContentType]):
    """Result of `chat.completions.parse()`: OpenAI's parsed completion plus the Interfaze extras."""

    precontext: Optional[List[Precontext]] = None
    reasoning: Optional[str] = None
    vcache: bool = False
    debug: Optional[Any] = None
