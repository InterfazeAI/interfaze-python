from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI, OpenAI

from ._chat import AsyncChat, Chat
from ._constants import (
    HEADER_ADMIN_KEY,
    HEADER_BYPASS_CACHE,
    HEADER_BYPASS_MOE,
    HEADER_SHOW_ADDITIONAL_INFO,
    INTERFAZE_BASE_URL,
)
from ._errors import InterfazeError
from ._tasks import AsyncTasks, Tasks


def _resolve_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("INTERFAZE_API_KEY")
    if not key:
        raise InterfazeError(
            "Missing API key. Pass Interfaze(api_key=...) or set the INTERFAZE_API_KEY environment variable."
        )
    return key


def _build_headers(
    default_headers: Optional[Dict[str, str]],
    show_additional_info: bool,
    bypass_moe: bool,
    bypass_cache: bool,
    admin_key: Optional[str],
) -> Dict[str, str]:
    headers: Dict[str, str] = dict(default_headers or {})
    if show_additional_info:
        headers[HEADER_SHOW_ADDITIONAL_INFO] = "true"
    if bypass_moe:
        headers[HEADER_BYPASS_MOE] = "true"
    if bypass_cache:
        headers[HEADER_BYPASS_CACHE] = "true"
    if admin_key:
        headers[HEADER_ADMIN_KEY] = admin_key
    return headers


class Interfaze:
    """Synchronous Interfaze client — a curated wrapper over ``openai.OpenAI``.

    Exposes the endpoints Interfaze implements (``chat.completions``, ``models``) plus task
    helpers (``tasks.*``). The underlying client is available at ``.openai``.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        show_additional_info: bool = False,
        bypass_moe: bool = False,
        bypass_cache: bool = False,
        admin_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.openai = OpenAI(
            api_key=_resolve_key(api_key),
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=_build_headers(
                default_headers, show_additional_info, bypass_moe, bypass_cache, admin_key
            ),
            **kwargs,
        )
        self.chat = Chat(self.openai)
        self.models = self.openai.models
        self.tasks = Tasks(self.chat.completions)


class AsyncInterfaze:
    """Asynchronous Interfaze client — a curated wrapper over ``openai.AsyncOpenAI``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        show_additional_info: bool = False,
        bypass_moe: bool = False,
        bypass_cache: bool = False,
        admin_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.openai = AsyncOpenAI(
            api_key=_resolve_key(api_key),
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=_build_headers(
                default_headers, show_additional_info, bypass_moe, bypass_cache, admin_key
            ),
            **kwargs,
        )
        self.chat = AsyncChat(self.openai)
        self.models = self.openai.models
        self.tasks = AsyncTasks(self.chat.completions)
