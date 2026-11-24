"""Credentials at the request boundary: keyless, environment-key and explicit-key requests."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import pytest
import respx
from conftest import BASIC, CHAT_URL, STREAM_CHUNKS, error_body, mock_status, sse_bytes

from interfaze import AsyncInterfaze, AuthenticationError, Interfaze, RateLimitError

MODELS_URL = "https://api.interfaze.ai/v1/models"
MODELS_LIST = {
    "object": "list",
    "data": [{"id": "interfaze-beta", "object": "model", "owned_by": "interfaze"}],
}
MESSAGES = [{"role": "user", "content": "x"}]

ENV_KEY = "interfaze-env-key"
EXPLICIT_KEY = "interfaze-explicit-key"
OPENAI_KEY = "openai-env-key"

# The API builds these error bodies in InterfazeAI/interfaze src/helpers/keylessGuard.ts and apiKeyAuth.ts.
SIGNUP_HINT = (
    "Get a free API key at https://interfaze.ai and send it as `Authorization: Bearer YOUR_API_KEY`."
)
KEYLESS_DAILY_LIMIT = error_body(
    f"Keyless daily limit of 50 requests reached. {SIGNUP_HINT}",
    "rate_limit_error",
    "keyless_daily_limit_exceeded",
)
INVALID_KEY = error_body("The API key provided is not valid", "permission_error", "insufficient_quota")

CLIENTS = [Interfaze, AsyncInterfaze]
CLIENT_IDS = ["sync", "async"]


def _chat(client: Any) -> Any:
    return client.chat.completions.create(messages=MESSAGES)


OPERATIONS: Dict[str, Callable[[Any], Any]] = {
    "chat": _chat,
    "stream": lambda client: client.chat.completions.stream(messages=MESSAGES).get_final_completion(),
    "raw_stream": lambda client: client.chat.completions.create(messages=MESSAGES, stream=True),
    "models": lambda client: client.models.list(),
}


async def _wait(value: Any) -> Any:
    return await value


def _call(client: Any, operation: Callable[[Any], Any]) -> Any:
    result = operation(client)
    return asyncio.run(_wait(result)) if inspect.isawaitable(result) else result


def _reply(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_bytes(STREAM_CHUNKS)
        )
    return httpx.Response(200, json=BASIC)


def _authorizations(router: respx.MockRouter) -> List[Optional[str]]:
    return [call.request.headers.get("authorization") for call in router.calls]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(CHAT_URL).mock(side_effect=_reply)
        router.get(MODELS_URL).mock(return_value=httpx.Response(200, json=MODELS_LIST))
        yield router


operations = pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
clients = pytest.mark.parametrize("client_cls", CLIENTS, ids=CLIENT_IDS)


@operations
@clients
def test_no_key_sends_no_authorization(api, client_cls, operation):
    _call(client_cls(), operation)
    assert _authorizations(api) == [None]


@operations
@clients
def test_no_key_ignores_openai_api_key(api, monkeypatch, client_cls, operation):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    _call(client_cls(), operation)
    assert _authorizations(api) == [None]


@operations
@clients
def test_env_key_authorizes_requests(api, monkeypatch, client_cls, operation):
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    _call(client_cls(), operation)
    assert _authorizations(api) == [f"Bearer {ENV_KEY}"]


@operations
@clients
def test_explicit_key_takes_precedence_over_env_key(api, monkeypatch, client_cls, operation):
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    _call(client_cls(api_key=EXPLICIT_KEY), operation)
    assert _authorizations(api) == [f"Bearer {EXPLICIT_KEY}"]


@clients
def test_no_key_survives_openai_with_options(api, monkeypatch, client_cls):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    _call(client_cls().openai.with_options(max_retries=0), OPERATIONS["models"])
    assert _authorizations(api) == [None]


@clients
@respx.mock
def test_rejected_key_raises_without_keyless_retry(client_cls):
    route = mock_status(401, INVALID_KEY)
    with pytest.raises(AuthenticationError) as exc_info:
        _call(client_cls(api_key=EXPLICIT_KEY), _chat)
    assert exc_info.value.status_code == 401
    assert exc_info.value.body == INVALID_KEY["error"]
    assert [call.request.headers.get("authorization") for call in route.calls] == [f"Bearer {EXPLICIT_KEY}"]


@clients
@respx.mock
def test_keyless_rate_limit_keeps_status_and_guidance(client_cls):
    route = mock_status(429, KEYLESS_DAILY_LIMIT)
    with pytest.raises(RateLimitError) as exc_info:
        _call(client_cls(max_retries=0), _chat)
    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "keyless_daily_limit_exceeded"
    assert exc_info.value.body == KEYLESS_DAILY_LIMIT["error"]
    assert SIGNUP_HINT in str(exc_info.value)
    assert route.calls.last.request.headers.get("authorization") is None
