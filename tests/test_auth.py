"""Credentials at the request boundary: keyless, environment-key and explicit-key requests."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import anyio
import httpx
import pytest
import respx
from conftest import STREAM_CHUNKS, completion, sse_bytes
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from interfaze import AsyncInterfaze, AuthenticationError, Interfaze, InternalServerError, RateLimitError
from interfaze._constants import INTERFAZE_BASE_URL

API_HOST = "api.interfaze.ai"
MESSAGES = [{"role": "user", "content": "x"}]
MODELS_LIST = {
    "object": "list",
    "data": [{"id": "interfaze-beta", "object": "model", "owned_by": "interfaze"}],
}
EMBEDDINGS = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
    "model": "interfaze-beta",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}
MODERATIONS = {"id": "modr-test", "model": "interfaze-beta", "results": []}
BATCHES = {"object": "list", "data": [], "has_more": False}
RESPONSE_BODIES = {
    "/organization/projects": BATCHES,
    "/models": MODELS_LIST,
    "/embeddings": EMBEDDINGS,
    "/moderations": MODERATIONS,
    "/batches": BATCHES,
}
WEATHER = {"city": "Tokyo", "temp_c": 21}

ENV_KEY = "interfaze-env-key"
EXPLICIT_KEY = "interfaze-explicit-key"
NEW_KEY = "interfaze-new-key"
OPENAI_KEY = "openai-env-key"
OPENAI_ADMIN_KEY = "openai-admin-key"
CALLER_AUTHORIZATION = "Bearer caller-token"

# Default retries make three attempts in total.
DEFAULT_ATTEMPTS = 3
SECONDS_LIMIT_FOR_NO_WAIT = 2.0
SECONDS_IN_AN_HOUR = 3600
KEYLESS_DAILY_LIMIT_REQUESTS = 1000
KEYLESS_INFLIGHT_LIMIT_REQUESTS = 5
# A tiny Retry-After keeps the retries fast and the attempt count exact.
INFLIGHT_RETRY_AFTER = "0.01"

# The API builds these error bodies in InterfazeAI/interfaze src/helpers/keylessGuard.ts and apiKeyAuth.ts.
SIGNUP_HINT = (
    "Get a free API key at https://interfaze.ai and send it as `Authorization: Bearer YOUR_API_KEY`."
)
REQUEST_ID = "req_keyless_test"


def _error(message: str, type_: str, code: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code, "request_id": REQUEST_ID}}


KEYLESS_DAILY_LIMIT = _error(
    f"Keyless daily limit of {KEYLESS_DAILY_LIMIT_REQUESTS} requests reached. {SIGNUP_HINT}",
    "rate_limit_error",
    "keyless_daily_limit_exceeded",
)
KEYLESS_INFLIGHT_LIMIT = _error(
    f"Too many keyless requests in progress (limit {KEYLESS_INFLIGHT_LIMIT_REQUESTS}). {SIGNUP_HINT}",
    "rate_limit_error",
    "keyless_inflight_limit_exceeded",
)
GENERIC_RATE_LIMIT = _error("Slow down", "rate_limit_error", "rate_limit_exceeded")
INVALID_KEY = _error("The API key provided is not valid", "permission_error", "insufficient_quota")


def _daily_headers() -> Dict[str, str]:
    return {
        "Retry-After": str(SECONDS_IN_AN_HOUR),
        "X-RateLimit-Limit": str(KEYLESS_DAILY_LIMIT_REQUESTS),
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int(time.time()) + SECONDS_IN_AN_HOUR),
    }


class _Weather(BaseModel):
    city: str
    temp_c: float


CLIENTS = [Interfaze, AsyncInterfaze]
CLIENT_IDS = ["sync", "async"]
OPERATIONS: Dict[str, Callable[[Any], Any]] = {
    "chat": lambda client: client.chat.completions.create(messages=MESSAGES),
    "stream": lambda client: client.chat.completions.stream(messages=MESSAGES).get_final_completion(),
    "raw_stream": lambda client: client.chat.completions.create(messages=MESSAGES, stream=True),
    "parse": lambda client: client.chat.completions.parse(messages=MESSAGES, response_format=_Weather),
    "models": lambda client: client.models.list(),
    "raw_embeddings": lambda client: client.openai.embeddings.create(model="interfaze-beta", input="x"),
    "raw_moderations": lambda client: client.openai.moderations.create(input="x"),
    "raw_batches": lambda client: client.openai.batches.list(),
}
CHAT = OPERATIONS["chat"]
RAW_STREAM = OPERATIONS["raw_stream"]
# A streamed async response has an unread body when the retry decision runs.
BODY_READABLE_OPERATIONS = ["chat", "parse", "models", "raw_embeddings", "raw_moderations", "raw_batches"]


def _list_models(openai: Any) -> Any:
    return openai.models.list()


async def _wait(value: Any) -> Any:
    return await value


def _call(client: Any, operation: Callable[[Any], Any]) -> Any:
    result = operation(client)
    return asyncio.run(_wait(result)) if inspect.isawaitable(result) else result


def _reply(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if not path.endswith("/chat/completions"):
        return httpx.Response(200, json=RESPONSE_BODIES[path.removeprefix("/v1")])
    if json.loads(request.content).get("stream"):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_bytes(STREAM_CHUNKS)
        )
    return httpx.Response(200, json=completion(json.dumps(WEATHER)))


def _authorizations(router: respx.MockRouter) -> List[Optional[str]]:
    return [call.request.headers.get("authorization") for call in router.calls]


def _sent_authorizations(router: respx.MockRouter) -> List[List[str]]:
    return [call.request.headers.get_list("authorization") for call in router.calls]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock_router:
        yield mock_router


@pytest.fixture
def api(router: respx.MockRouter) -> respx.MockRouter:
    router.route(host=API_HOST).mock(side_effect=_reply)
    return router


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(anyio, "sleep", no_sleep)


def _fail(router: respx.MockRouter, status: int, body: Dict[str, Any], headers: Dict[str, str]) -> None:
    router.route(host=API_HOST).mock(return_value=httpx.Response(status, json=body, headers=headers))


operations = pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
# openai 2.34 and later have the admin resource, which reads OPENAI_ADMIN_KEY.
ADMIN = pytest.param(
    lambda client: client.openai.admin.organization.projects.list(),
    id="admin",
    marks=pytest.mark.skipif(not hasattr(OpenAI, "admin"), reason="This openai has no admin resource."),
)
credential_operations = pytest.mark.parametrize(
    "operation", [*(pytest.param(op, id=name) for name, op in OPERATIONS.items()), ADMIN]
)
clients = pytest.mark.parametrize("client_cls", CLIENTS, ids=CLIENT_IDS)


@operations
@clients
def test_no_key_sends_no_authorization(api, client_cls, operation):
    _call(client_cls(), operation)
    assert _authorizations(api) == [None]


@credential_operations
@clients
def test_no_key_ignores_openai_credentials(api, monkeypatch, client_cls, operation):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", OPENAI_ADMIN_KEY)
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


# Each case: explicit api_key, INTERFAZE_API_KEY, the Authorization header the request carries.
KEY_CASES = {
    "padded-explicit-key": (f" {EXPLICIT_KEY}\n", "", f"Bearer {EXPLICIT_KEY}"),
    "padded-env-key": (None, f" {ENV_KEY}\n", f"Bearer {ENV_KEY}"),
    "empty-explicit-key-is-keyless": ("", "", None),
    "space-explicit-key-is-keyless": (" ", "", None),
    "newline-explicit-key-is-keyless": (" \n", "", None),
    "empty-explicit-key-uses-env-key": ("", ENV_KEY, f"Bearer {ENV_KEY}"),
    "blank-explicit-key-uses-env-key": ("   ", ENV_KEY, f"Bearer {ENV_KEY}"),
    "newline-explicit-key-uses-env-key": (" \n", ENV_KEY, f"Bearer {ENV_KEY}"),
    "blank-env-key-is-keyless": (None, " \n", None),
}


@pytest.mark.parametrize("api_key,env_key,authorization", KEY_CASES.values(), ids=KEY_CASES.keys())
@clients
def test_key_is_trimmed_or_blank(api, monkeypatch, client_cls, api_key, env_key, authorization):
    monkeypatch.setenv("INTERFAZE_API_KEY", env_key)
    _call(client_cls(api_key=api_key), CHAT)
    assert _authorizations(api) == [authorization]


def _sync_key() -> str:
    return EXPLICIT_KEY


async def _async_key() -> str:
    return EXPLICIT_KEY


KEY_CALLABLES = {"sync": (Interfaze, _sync_key), "async": (AsyncInterfaze, _async_key)}


@operations
@pytest.mark.parametrize("client_cls,key_callable", KEY_CALLABLES.values(), ids=KEY_CALLABLES.keys())
def test_key_callable_authorizes_requests(api, client_cls, key_callable, operation):
    _call(client_cls(api_key=key_callable), operation)
    assert _authorizations(api) == [f"Bearer {EXPLICIT_KEY}"]


@pytest.mark.parametrize("client_cls,key_callable", KEY_CALLABLES.values(), ids=KEY_CALLABLES.keys())
def test_key_callable_takes_precedence_over_env_key(api, monkeypatch, client_cls, key_callable):
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    _call(client_cls(api_key=key_callable), CHAT)
    assert _authorizations(api) == [f"Bearer {EXPLICIT_KEY}"]


# Each case: how to derive a client from client.openai.
DERIVED = {
    "with_options": lambda openai, **kwargs: openai.with_options(**kwargs),
    "copy": lambda openai, **kwargs: openai.copy(**kwargs),
}
derived = pytest.mark.parametrize("derive", DERIVED.values(), ids=DERIVED.keys())


@derived
@clients
def test_no_key_survives_derived_client(api, monkeypatch, client_cls, derive):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    _call(derive(client_cls().openai, max_retries=0), _list_models)
    assert _authorizations(api) == [None]


@derived
@clients
def test_derived_client_sends_new_key_from_keyless_client(api, client_cls, derive):
    _call(derive(client_cls().openai, api_key=NEW_KEY), _list_models)
    assert _authorizations(api) == [f"Bearer {NEW_KEY}"]


@derived
@clients
def test_derived_client_sends_new_key_over_old_key(api, client_cls, derive):
    _call(derive(client_cls(api_key=EXPLICIT_KEY).openai, api_key=NEW_KEY), _list_models)
    assert _authorizations(api) == [f"Bearer {NEW_KEY}"]


@derived
@clients
def test_derived_client_keeps_key(api, client_cls, derive):
    _call(derive(client_cls(api_key=EXPLICIT_KEY).openai, max_retries=0), _list_models)
    assert _authorizations(api) == [f"Bearer {EXPLICIT_KEY}"]


# Each case: the key the client holds, and the spelling of the Authorization header the caller sets.
CALLER_HEADER_CASES = {
    "keyless": (None, "Authorization"),
    "keyless-lowercase": (None, "authorization"),
    "keyless-uppercase": (None, "AUTHORIZATION"),
    "keyed": (EXPLICIT_KEY, "Authorization"),
    "keyed-lowercase": (EXPLICIT_KEY, "authorization"),
    "keyed-uppercase": (EXPLICIT_KEY, "AUTHORIZATION"),
}
caller_header = pytest.mark.parametrize(
    "api_key,header_name", CALLER_HEADER_CASES.values(), ids=CALLER_HEADER_CASES.keys()
)


@operations
@caller_header
@clients
def test_caller_default_header_is_sent_once(api, client_cls, api_key, header_name, operation):
    client = client_cls(api_key=api_key, default_headers={header_name: CALLER_AUTHORIZATION})
    _call(client, operation)
    assert _sent_authorizations(api) == [[CALLER_AUTHORIZATION]]


@caller_header
@clients
def test_caller_request_header_is_sent_once(api, client_cls, api_key, header_name):
    client = client_cls(api_key=api_key)
    headers = {header_name: CALLER_AUTHORIZATION}
    _call(client, lambda c: c.chat.completions.create(messages=MESSAGES, extra_headers=headers))
    _call(client, lambda c: c.openai.models.list(extra_headers=headers))
    assert _sent_authorizations(api) == [[CALLER_AUTHORIZATION], [CALLER_AUTHORIZATION]]


@caller_header
@clients
def test_caller_default_header_survives_derived_client(api, client_cls, api_key, header_name):
    client = client_cls(api_key=api_key, default_headers={header_name: CALLER_AUTHORIZATION})
    _call(client.openai.with_options(max_retries=0), _list_models)
    assert _sent_authorizations(api) == [[CALLER_AUTHORIZATION]]


@operations
@clients
def test_rejected_key_raises_after_one_request(router, client_cls, operation):
    _fail(router, 401, INVALID_KEY, {})
    with pytest.raises(AuthenticationError) as exc_info:
        _call(client_cls(api_key=EXPLICIT_KEY), operation)
    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "insufficient_quota"
    assert exc_info.value.body == INVALID_KEY["error"]
    assert _authorizations(router) == [f"Bearer {EXPLICIT_KEY}"]


@operations
@clients
def test_keyless_401_raises_after_one_request(router, client_cls, operation):
    _fail(router, 401, INVALID_KEY, {})
    with pytest.raises(AuthenticationError) as exc_info:
        _call(client_cls(), operation)
    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "insufficient_quota"
    assert exc_info.value.body == INVALID_KEY["error"]
    assert _authorizations(router) == [None]


@operations
@clients
def test_keyless_daily_limit_keeps_status_and_guidance(router, client_cls, operation):
    _fail(router, 429, KEYLESS_DAILY_LIMIT, _daily_headers())
    with pytest.raises(RateLimitError) as exc_info:
        _call(client_cls(max_retries=0), operation)
    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "keyless_daily_limit_exceeded"
    assert exc_info.value.body == KEYLESS_DAILY_LIMIT["error"]
    assert SIGNUP_HINT in str(exc_info.value)
    assert _authorizations(router) == [None]


@operations
@clients
def test_keyless_daily_limit_raises_after_one_request_with_default_retries(router, client_cls, operation):
    _fail(router, 429, KEYLESS_DAILY_LIMIT, _daily_headers())
    started = time.monotonic()
    with pytest.raises(RateLimitError) as exc_info:
        _call(client_cls(), operation)
    assert time.monotonic() - started < SECONDS_LIMIT_FOR_NO_WAIT
    assert router.calls.call_count == 1
    assert exc_info.value.status_code == 429
    assert exc_info.value.code == "keyless_daily_limit_exceeded"
    assert exc_info.value.body == KEYLESS_DAILY_LIMIT["error"]
    assert SIGNUP_HINT in str(exc_info.value)


@pytest.mark.parametrize("operation_name", BODY_READABLE_OPERATIONS)
@clients
def test_keyless_daily_limit_is_detected_by_body_code_without_retry_after(router, client_cls, operation_name):
    _fail(router, 429, KEYLESS_DAILY_LIMIT, {})
    with pytest.raises(RateLimitError):
        _call(client_cls(), OPERATIONS[operation_name])
    assert router.calls.call_count == 1


def test_sync_stream_reads_the_body_to_detect_the_daily_limit(router):
    _fail(router, 429, KEYLESS_DAILY_LIMIT, {})
    with pytest.raises(RateLimitError) as exc_info:
        _call(Interfaze(), RAW_STREAM)
    assert router.calls.call_count == 1
    assert exc_info.value.body == KEYLESS_DAILY_LIMIT["error"]


class _BrokenBody(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b'{"error": {"code": "rate'
        raise httpx.ReadError("connection dropped")


# Each case: the client options, keyed and keyless.
BROKEN_BODY_CLIENTS = {"keyless": {}, "keyed": {"api_key": EXPLICIT_KEY}}


@pytest.mark.parametrize("options", BROKEN_BODY_CLIENTS.values(), ids=BROKEN_BODY_CLIENTS.keys())
def test_sync_stream_429_with_a_broken_body_is_retried(router, options):
    router.route(host=API_HOST).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": INFLIGHT_RETRY_AFTER}, stream=_BrokenBody()),
            httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=sse_bytes(STREAM_CHUNKS)
            ),
        ]
    )
    _call(Interfaze(**options), RAW_STREAM)
    assert router.calls.call_count == 2


def test_async_stream_with_a_long_retry_after_is_not_retried(router):
    # The body of a streamed async response is unread, so a Retry-After above 60 seconds decides.
    _fail(router, 429, GENERIC_RATE_LIMIT, _daily_headers())
    with pytest.raises(RateLimitError):
        _call(AsyncInterfaze(), RAW_STREAM)
    assert router.calls.call_count == 1


# Each case: the client options that authenticate the client.
AUTHENTICATED_CLIENTS = {
    "keyed": {"api_key": EXPLICIT_KEY},
    "caller-header": {"default_headers": {"Authorization": CALLER_AUTHORIZATION}},
}


BARE_CLIENTS = {Interfaze: OpenAI, AsyncInterfaze: AsyncOpenAI}


def _bare_stream(client: Any) -> Any:
    return client.chat.completions.create(model="interfaze-beta", messages=MESSAGES, stream=True)


@pytest.mark.parametrize("options", AUTHENTICATED_CLIENTS.values(), ids=AUTHENTICATED_CLIENTS.keys())
@clients
def test_long_retry_after_with_authentication_keeps_the_openai_retries(
    router, no_retry_sleep, client_cls, options
):
    # The attempt count of a plain openai client depends on the openai version, so compare against it.
    _fail(router, 429, GENERIC_RATE_LIMIT, _daily_headers())
    with pytest.raises(RateLimitError):
        _call(client_cls(**options), RAW_STREAM)
    attempts = router.calls.call_count
    bare = BARE_CLIENTS[client_cls](
        api_key=options.get("api_key", EXPLICIT_KEY),
        base_url=INTERFAZE_BASE_URL,
        default_headers=options.get("default_headers"),
    )
    with pytest.raises(RateLimitError):
        _call(bare, _bare_stream)
    assert router.calls.call_count - attempts == attempts


# Each case: a 429 response content that carries no keyless daily limit code.
UNRECOGNIZED_BODIES = {
    "text": b"upstream busy",
    "json-list": b"[]",
    "error-text": b'{"error": "busy"}',
    "error-without-code": b'{"error": {"message": "busy"}}',
    "error-numeric-code": b'{"error": {"code": 429}}',
}


@pytest.mark.parametrize("content", UNRECOGNIZED_BODIES.values(), ids=UNRECOGNIZED_BODIES.keys())
@clients
def test_429_with_an_unrecognized_body_is_retried(router, no_retry_sleep, client_cls, content):
    router.route(host=API_HOST).mock(return_value=httpx.Response(429, content=content))
    with pytest.raises(RateLimitError):
        _call(client_cls(), CHAT)
    assert router.calls.call_count == DEFAULT_ATTEMPTS


def test_async_stream_with_an_unparsable_retry_after_is_retried(router, no_retry_sleep):
    _fail(router, 429, GENERIC_RATE_LIMIT, {"Retry-After": "soon"})
    with pytest.raises(RateLimitError):
        _call(AsyncInterfaze(), RAW_STREAM)
    assert router.calls.call_count == DEFAULT_ATTEMPTS


@operations
@clients
def test_keyless_server_error_is_still_retried(router, no_retry_sleep, client_cls, operation):
    _fail(router, 500, GENERIC_RATE_LIMIT, {})
    with pytest.raises(InternalServerError):
        _call(client_cls(), operation)
    assert router.calls.call_count == DEFAULT_ATTEMPTS


@clients
def test_header_validation_accepts_any_arguments(client_cls):
    assert client_cls().openai._validate_headers({}, {}, "extra", flag=True) is None


@operations
@clients
def test_keyless_inflight_limit_is_still_retried(router, client_cls, operation):
    _fail(router, 429, KEYLESS_INFLIGHT_LIMIT, {"Retry-After": INFLIGHT_RETRY_AFTER})
    with pytest.raises(RateLimitError) as exc_info:
        _call(client_cls(), operation)
    assert router.calls.call_count == DEFAULT_ATTEMPTS
    assert exc_info.value.code == "keyless_inflight_limit_exceeded"
    assert exc_info.value.body == KEYLESS_INFLIGHT_LIMIT["error"]
    assert SIGNUP_HINT in str(exc_info.value)


@clients
def test_keyless_inflight_limit_with_the_api_retry_after_is_retried(router, no_retry_sleep, client_cls):
    _fail(router, 429, KEYLESS_INFLIGHT_LIMIT, {"Retry-After": str(KEYLESS_INFLIGHT_LIMIT_REQUESTS)})
    with pytest.raises(RateLimitError):
        _call(client_cls(), RAW_STREAM)
    assert router.calls.call_count == DEFAULT_ATTEMPTS
