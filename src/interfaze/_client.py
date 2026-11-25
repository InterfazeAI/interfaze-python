from __future__ import annotations

import os
import warnings
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from openai import AsyncOpenAI, OpenAI

from ._chat import AsyncChat, Chat
from ._constants import (
    DEFAULT_TIMEOUT,
    HEADER_ADMIN_KEY,
    HEADER_BYPASS_CACHE,
    HEADER_BYPASS_MOA,
    HEADER_SHOW_ADDITIONAL_INFO,
    INTERFAZE_BASE_URL,
    KEYLESS_WARNING,
)
from ._tasks import AsyncTasks, Tasks


_KeyProvider = Callable[[], Any]

_keyless_warned = False


# openai calls a key provider before each request, so the wrapper trims the key each time.
def _trimmed(provider: _KeyProvider) -> _KeyProvider:
    return lambda: provider().strip()


def _async_trimmed(provider: _KeyProvider) -> _KeyProvider:
    async def trimmed() -> str:
        return str((await provider()).strip())

    return trimmed


# Trims each string key, because an HTTP header value cannot end with a space or a newline.
def _resolve_key(
    api_key: Union[str, _KeyProvider, None], trim_provider: Callable[[_KeyProvider], _KeyProvider]
) -> Union[str, _KeyProvider, None]:
    if callable(api_key):
        return trim_provider(api_key)
    return (api_key or "").strip() or os.environ.get("INTERFAZE_API_KEY", "").strip() or None


# A caller Authorization header counts as a key.
def _has_authorization(headers: Dict[str, str]) -> bool:
    return any(name.lower() == "authorization" for name in headers)


# Warns once per process; stacklevel=3 points at the code that built the client.
def _warn_if_keyless(key: Union[str, _KeyProvider, None], headers: Dict[str, str]) -> None:
    global _keyless_warned
    if key or _has_authorization(headers) or _keyless_warned:
        return
    _keyless_warned = True
    warnings.warn(KEYLESS_WARNING, UserWarning, stacklevel=3)


# A key provider that returns "" stops OpenAI() from reading OPENAI_API_KEY and sends no Authorization header.
def _no_key() -> str:
    return ""


async def _async_no_key() -> str:
    return ""


# openai>=2.34 refuses requests with no Authorization header; Interfaze allows keyless.
class InterfazeOpenAI(OpenAI):
    # openai>=2.34 reads OPENAI_ADMIN_KEY; keep that credential off Interfaze requests.
    @property
    def _admin_api_key_auth(self) -> Dict[str, str]:
        return {}

    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        return None


class AsyncInterfazeOpenAI(AsyncOpenAI):
    @property
    def _admin_api_key_auth(self) -> Dict[str, str]:
        return {}

    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        return None


def _build_headers(
    default_headers: Optional[Dict[str, str]],
    show_additional_info: bool,
    bypass_moa: bool,
    bypass_cache: bool,
    admin_key: Optional[str],
) -> Dict[str, str]:
    headers: Dict[str, str] = dict(default_headers or {})
    if show_additional_info:
        headers[HEADER_SHOW_ADDITIONAL_INFO] = "true"
    if bypass_moa:
        headers[HEADER_BYPASS_MOA] = "true"
    if bypass_cache:
        headers[HEADER_BYPASS_CACHE] = "true"
    if admin_key:
        headers[HEADER_ADMIN_KEY] = admin_key
    return headers


# OpenAI() fills `organization` / `project` from OPENAI_ORG_ID / OPENAI_PROJECT_ID and sends them
# as OpenAI-Organization / OpenAI-Project headers. Those are OpenAI account ids, so keep them off
# Interfaze requests unless they were passed in explicitly.
# This only covers the client built here: one derived via client.openai.with_options(...) or
# .copy() goes back through the constructor with organization/project None, so it reads the
# env vars again and sends them.
def _drop_openai_env_ids(client: Union[OpenAI, AsyncOpenAI], kwargs: Dict[str, Any]) -> None:
    client.organization = kwargs.get("organization")
    client.project = kwargs.get("project")


class Interfaze:
    def __init__(
        self,
        *,
        api_key: Union[str, Callable[[], str], None] = None,
        base_url: Optional[str] = None,
        show_additional_info: bool = False,
        bypass_moa: bool = False,
        bypass_cache: bool = False,
        admin_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        key = _resolve_key(api_key, _trimmed)
        headers = _build_headers(default_headers, show_additional_info, bypass_moa, bypass_cache, admin_key)
        _warn_if_keyless(key, headers)
        self.openai: OpenAI = InterfazeOpenAI(
            api_key=key or _no_key,
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=headers,
            **kwargs,
        )
        _drop_openai_env_ids(self.openai, kwargs)
        self.chat = Chat(self.openai)
        self.models = self.openai.models
        self.tasks = Tasks(self.chat.completions)


class AsyncInterfaze:
    def __init__(
        self,
        *,
        api_key: Union[str, Callable[[], Awaitable[str]], None] = None,
        base_url: Optional[str] = None,
        show_additional_info: bool = False,
        bypass_moa: bool = False,
        bypass_cache: bool = False,
        admin_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        key = _resolve_key(api_key, _async_trimmed)
        headers = _build_headers(default_headers, show_additional_info, bypass_moa, bypass_cache, admin_key)
        _warn_if_keyless(key, headers)
        self.openai: AsyncOpenAI = AsyncInterfazeOpenAI(
            api_key=key or _async_no_key,
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=headers,
            **kwargs,
        )
        _drop_openai_env_ids(self.openai, kwargs)
        self.chat = AsyncChat(self.openai)
        self.models = self.openai.models
        self.tasks = AsyncTasks(self.chat.completions)
