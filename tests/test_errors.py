from __future__ import annotations

import asyncio
import time

import anyio
import httpx
import pytest
import respx
from conftest import BASIC, CHAT_URL, error_body, mock_status

from interfaze import (
    APIStatusError,
    APITimeoutError,
    AsyncInterfaze,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    Interfaze,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

STATUS_MAP = [
    (
        400,
        BadRequestError,
        error_body(
            "Field 'temperature': Too big: expected number to be <=1",
            "invalid_request_error",
            "invalid_request",
        ),
    ),
    (
        401,
        AuthenticationError,
        error_body("Invalid API key provided", "authentication_error", "invalid_api_key"),
    ),
    (
        403,
        PermissionDeniedError,
        error_body("You do not have access to this resource", "permission_error", "forbidden"),
    ),
    (
        404,
        NotFoundError,
        error_body("The requested model does not exist", "not_found_error", "model_not_found"),
    ),
    (
        429,
        RateLimitError,
        error_body("Rate limit exceeded, please slow down", "rate_limit_error", "rate_limit_exceeded"),
    ),
    (500, InternalServerError, error_body("An internal error occurred", "server_error", "internal_error")),
    (
        503,
        InternalServerError,
        error_body("The service is temporarily unavailable", "server_error", "service_unavailable"),
    ),
]
STATUS_IDS = [str(row[0]) for row in STATUS_MAP]

RATE_LIMITED = error_body("Rate limit exceeded", "rate_limit_error", "rate_limit_exceeded")
SERVER_ERROR = error_body("Internal error", "server_error", "internal_error")


@pytest.mark.parametrize("status,exc_type,body", STATUS_MAP, ids=STATUS_IDS)
@respx.mock
def test_status_maps_to_exception(status, exc_type, body):
    mock_status(status, body)
    with pytest.raises(exc_type) as exc_info:
        Interfaze(api_key="t", max_retries=0).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )
    assert isinstance(exc_info.value, APIStatusError)
    assert exc_info.value.status_code == status


@respx.mock
def test_bad_request_error_body_reaches_caller():
    """The mapped exception must carry the actual server-provided error details, not just
    the right type — callers rely on `.body["code"]` to branch on specific failures."""
    body = STATUS_MAP[0][2]
    mock_status(400, body)
    with pytest.raises(BadRequestError) as exc_info:
        Interfaze(api_key="t", max_retries=0).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )
    assert exc_info.value.body["code"] == "invalid_request"
    assert "temperature" in exc_info.value.body["message"]


@pytest.mark.parametrize("status,exc_type,body", STATUS_MAP, ids=STATUS_IDS)
@respx.mock
def test_status_maps_to_exception_async(status, exc_type, body):
    mock_status(status, body)

    async def go():
        await AsyncInterfaze(api_key="t", max_retries=0).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )

    with pytest.raises(exc_type) as exc_info:
        asyncio.run(go())
    assert exc_info.value.status_code == status


@respx.mock
def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json=RATE_LIMITED),
            httpx.Response(429, json=RATE_LIMITED),
            httpx.Response(200, json=BASIC),
        ]
    )
    r = Interfaze(api_key="t", max_retries=2).chat.completions.create(
        messages=[{"role": "user", "content": "x"}]
    )
    assert r.choices[0].message.content == "Hi!"
    assert route.calls.call_count == 3


@respx.mock
def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    route = mock_status(500, SERVER_ERROR)
    with pytest.raises(InternalServerError):
        Interfaze(api_key="t", max_retries=2).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )
    assert route.calls.call_count == 3


@respx.mock
def test_async_retries_then_succeeds(monkeypatch):
    async def noop_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(anyio, "sleep", noop_sleep)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json=RATE_LIMITED),
            httpx.Response(429, json=RATE_LIMITED),
            httpx.Response(200, json=BASIC),
        ]
    )

    async def go():
        return await AsyncInterfaze(api_key="t", max_retries=2).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )

    r = asyncio.run(go())
    assert r.choices[0].message.content == "Hi!"
    assert route.calls.call_count == 3


@respx.mock
def test_async_retries_exhausted_raises(monkeypatch):
    async def noop_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(anyio, "sleep", noop_sleep)
    route = mock_status(500, SERVER_ERROR)

    async def go():
        await AsyncInterfaze(api_key="t", max_retries=2).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )

    with pytest.raises(InternalServerError):
        asyncio.run(go())
    assert route.calls.call_count == 3


@respx.mock
def test_timeout_raises_api_timeout_error():
    respx.post(CHAT_URL).mock(side_effect=httpx.TimeoutException("timed out"))
    with pytest.raises(APITimeoutError):
        Interfaze(api_key="t", max_retries=0).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )


@respx.mock
def test_async_timeout_raises_api_timeout_error():
    respx.post(CHAT_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    async def go():
        await AsyncInterfaze(api_key="t", max_retries=0).chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )

    with pytest.raises(APITimeoutError):
        asyncio.run(go())
