from __future__ import annotations

import contextlib
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import httpx
from openai import AsyncOpenAI, OpenAI

from ._chat import AsyncChat, Chat
from ._constants import (
    DEFAULT_TIMEOUT,
    HEADER_ADMIN_KEY,
    HEADER_BYPASS_CACHE,
    HEADER_BYPASS_MOA,
    HEADER_SHOW_ADDITIONAL_INFO,
    INTERFAZE_BASE_URL,
    KEYLESS_DAILY_LIMIT_CODE,
    MAX_HONORED_RETRY_AFTER_SECONDS,
    NO_RETRY_AFTER_SECONDS,
    SINGLE_HEADER_VALUE,
)
from ._tasks import AsyncTasks, Tasks


_KeyProvider = Callable[[], Any]


# Returns the explicit key, else INTERFAZE_API_KEY, else None for a keyless client. Trims each string
# key, because an HTTP header value cannot end with a space or a newline. A key provider passes
# through untouched, because openai calls it before each request.
def _resolve_key(api_key: Union[str, _KeyProvider, None]) -> Union[str, _KeyProvider, None]:
    if callable(api_key):
        return api_key
    return (api_key or "").strip() or os.environ.get("INTERFAZE_API_KEY", "").strip() or None


# A keyless client gives OpenAI() a key provider that returns "". The provider stops OpenAI()
# from reading OPENAI_API_KEY and passes its credential check (openai 2.34 and later). OpenAI
# sends no Authorization header for an empty key, and with_options() and copy() keep the provider.
def _no_key() -> str:
    return ""


async def _async_no_key() -> str:
    return ""


def _error_code(body: object) -> Optional[str]:
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return NO_RETRY_AFTER_SECONDS


# Returns True for the keyless daily limit, which a retry cannot clear. A request with an
# Authorization header is never keyless, so keyed clients retry like openai. The error code
# identifies the limit when the body is readable. An unread body leaves the Retry-After header:
# openai waits for at most 60 seconds, so a longer wait marks the limit.
def _is_keyless_daily_limit(response: httpx.Response) -> bool:
    if response.status_code != 429 or "authorization" in response.request.headers:
        return False
    try:
        return _error_code(response.json()) == KEYLESS_DAILY_LIMIT_CODE
    except httpx.ResponseNotRead:
        return _retry_after_seconds(response) > MAX_HONORED_RETRY_AFTER_SECONDS
    except ValueError:
        return False


# openai joins the key and each caller header as separate dict keys, so an Authorization header with
# another letter case reaches the request next to the key. The caller header comes last and wins.
def _keep_last_authorization(headers: httpx.Headers) -> httpx.Headers:
    values = headers.get_list("authorization")
    if len(values) > SINGLE_HEADER_VALUE:
        headers["authorization"] = values.pop()
    return headers


# openai 2.34 and later raise before they send a request with no Authorization header.
# Interfaze accepts server-side requests with no key, so these clients skip that check.
# The *args signatures keep the overrides working when openai changes its private signatures.
# openai 2.34 and later read OPENAI_ADMIN_KEY for client.openai.admin.*. The empty _admin_api_key_auth
# keeps that OpenAI credential off Interfaze requests, and older openai versions never read it.
class InterfazeOpenAI(OpenAI):
    @property
    def _admin_api_key_auth(self) -> Dict[str, str]:
        return {}

    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _build_headers(self, *args: Any, **kwargs: Any) -> httpx.Headers:
        return _keep_last_authorization(super()._build_headers(*args, **kwargs))

    # A streamed sync response has an unread body here, so read it to find the error code. A body
    # that breaks while it is read leaves the Retry-After header to decide.
    def _should_retry(self, response: httpx.Response) -> bool:
        if response.status_code == 429 and not response.is_closed:
            with contextlib.suppress(httpx.HTTPError):
                response.read()
        return not _is_keyless_daily_limit(response) and super()._should_retry(response)


class AsyncInterfazeOpenAI(AsyncOpenAI):
    @property
    def _admin_api_key_auth(self) -> Dict[str, str]:
        return {}

    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _build_headers(self, *args: Any, **kwargs: Any) -> httpx.Headers:
        return _keep_last_authorization(super()._build_headers(*args, **kwargs))

    # The keyless daily limit check is a guess from the Retry-After header, because an async
    # stream body is not readable yet in this sync hook.
    def _should_retry(self, response: httpx.Response) -> bool:
        return not _is_keyless_daily_limit(response) and super()._should_retry(response)


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
        self.openai: OpenAI = InterfazeOpenAI(
            api_key=_resolve_key(api_key) or _no_key,
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=_build_headers(
                default_headers, show_additional_info, bypass_moa, bypass_cache, admin_key
            ),
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
        self.openai: AsyncOpenAI = AsyncInterfazeOpenAI(
            api_key=_resolve_key(api_key) or _async_no_key,
            base_url=base_url or INTERFAZE_BASE_URL,
            default_headers=_build_headers(
                default_headers, show_additional_info, bypass_moa, bypass_cache, admin_key
            ),
            **kwargs,
        )
        _drop_openai_env_ids(self.openai, kwargs)
        self.chat = AsyncChat(self.openai)
        self.models = self.openai.models
        self.tasks = AsyncTasks(self.chat.completions)
