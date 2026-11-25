"""Credentials at the request boundary: keyless, environment-key and explicit-key requests."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
import warnings
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
import pytest
import respx
from conftest import completion
from openai import OpenAI

import interfaze._client as client_module
from interfaze import AsyncInterfaze, Interfaze, RateLimitError

API_HOST = "api.interfaze.ai"
MESSAGES = [{"role": "user", "content": "x"}]
MODELS_LIST = {
    "object": "list",
    "data": [{"id": "interfaze-beta", "object": "model", "owned_by": "interfaze"}],
}
PROJECTS_LIST = {"object": "list", "data": [], "has_more": False}
RESPONSE_BODIES = {"/models": MODELS_LIST, "/organization/projects": PROJECTS_LIST}
WEATHER = {"city": "Tokyo", "temp_c": 21}

ENV_KEY = "interfaze-env-key"
EXPLICIT_KEY = "interfaze-explicit-key"
NEW_KEY = "interfaze-new-key"
OPENAI_KEY = "openai-env-key"
OPENAI_ADMIN_KEY = "openai-admin-key"

SECONDS_IN_AN_HOUR = 3600
KEYLESS_DAILY_LIMIT_REQUESTS = 1000
KEYLESS_WARNING = (
    "Interfaze SDK is running in keyless mode. "
    "For higher rate limits, get an API key at https://interfaze.ai/dashboard."
)

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


# The API sends these headers with the keyless daily limit 429.
def _daily_headers() -> Dict[str, str]:
    return {
        "Retry-After": str(SECONDS_IN_AN_HOUR),
        "X-RateLimit-Limit": str(KEYLESS_DAILY_LIMIT_REQUESTS),
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int(time.time()) + SECONDS_IN_AN_HOUR),
        "x-should-retry": "false",
    }


CLIENTS = [Interfaze, AsyncInterfaze]
CLIENT_IDS = ["sync", "async"]


def _chat(client: Any) -> Any:
    return client.chat.completions.create(messages=MESSAGES)


def _list_models(openai: Any) -> Any:
    return openai.models.list()


async def _wait(value: Any) -> Any:
    return await value


def _call(client: Any, operation: Callable[[Any], Any]) -> Any:
    result = operation(client)
    return asyncio.run(_wait(result)) if inspect.isawaitable(result) else result


def _reply(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/chat/completions"):
        return httpx.Response(200, json=completion(json.dumps(WEATHER)))
    return httpx.Response(200, json=RESPONSE_BODIES[path.removeprefix("/v1")])


def _authorizations(router: respx.MockRouter) -> List[Optional[str]]:
    return [call.request.headers.get("authorization") for call in router.calls]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(client_module, "_keyless_warned", False)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock_router:
        yield mock_router


@pytest.fixture
def api(router: respx.MockRouter) -> respx.MockRouter:
    router.route(host=API_HOST).mock(side_effect=_reply)
    return router


def _fail(router: respx.MockRouter, status: int, body: Dict[str, Any], headers: Dict[str, str]) -> None:
    router.route(host=API_HOST).mock(return_value=httpx.Response(status, json=body, headers=headers))


# openai 2.34 and later have the admin resource, which reads OPENAI_ADMIN_KEY.
ADMIN = pytest.param(
    lambda client: client.openai.admin.organization.projects.list(),
    id="admin",
    marks=pytest.mark.skipif(not hasattr(OpenAI, "admin"), reason="This openai has no admin resource."),
)
credential_operations = pytest.mark.parametrize("operation", [pytest.param(_chat, id="chat"), ADMIN])
clients = pytest.mark.parametrize("client_cls", CLIENTS, ids=CLIENT_IDS)


@credential_operations
@clients
def test_no_key_ignores_openai_credentials(api, monkeypatch, client_cls, operation):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", OPENAI_ADMIN_KEY)
    _call(client_cls(), operation)
    assert _authorizations(api) == [None]


# Each case: explicit api_key, INTERFAZE_API_KEY, the Authorization header the request carries.
KEY_CASES = {
    "env-key": (None, ENV_KEY, f"Bearer {ENV_KEY}"),
    "explicit-key-over-env-key": (EXPLICIT_KEY, ENV_KEY, f"Bearer {EXPLICIT_KEY}"),
    "padded-key-is-trimmed": (f" {EXPLICIT_KEY}\n", "", f"Bearer {EXPLICIT_KEY}"),
    "blank-explicit-key-uses-env-key": (" \n", ENV_KEY, f"Bearer {ENV_KEY}"),
    "blank-env-key-is-keyless": (None, " \n", None),
}


@pytest.mark.parametrize("api_key,env_key,authorization", KEY_CASES.values(), ids=KEY_CASES.keys())
@clients
def test_request_sends_the_resolved_key(api, monkeypatch, client_cls, api_key, env_key, authorization):
    monkeypatch.setenv("INTERFAZE_API_KEY", env_key)
    _call(client_cls(api_key=api_key), _chat)
    assert _authorizations(api) == [authorization]


def _sync_key() -> str:
    return EXPLICIT_KEY


async def _async_key() -> str:
    return EXPLICIT_KEY


KEY_CALLABLES = {"sync": (Interfaze, _sync_key), "async": (AsyncInterfaze, _async_key)}


@pytest.mark.parametrize("client_cls,key_callable", KEY_CALLABLES.values(), ids=KEY_CALLABLES.keys())
def test_key_callable_takes_precedence_over_env_key(api, monkeypatch, client_cls, key_callable):
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    _call(client_cls(api_key=key_callable), _chat)
    assert _authorizations(api) == [f"Bearer {EXPLICIT_KEY}"]


# Each case: the value a key callable returns, and the Authorization header the request carries.
CALLABLE_KEY = "interfaze-callable-key"
KEY_CALLABLE_RESULTS = {
    "padded-key-is-trimmed": (f" {CALLABLE_KEY}\n", f"Bearer {CALLABLE_KEY}"),
    "blank-key-is-keyless": ("  ", None),
}
key_callable_results = pytest.mark.parametrize(
    "result,authorization", KEY_CALLABLE_RESULTS.values(), ids=KEY_CALLABLE_RESULTS.keys()
)


def _key_callable(client_cls: Any, result: str) -> Callable[[], Any]:
    async def async_key() -> str:
        return result

    return async_key if client_cls is AsyncInterfaze else lambda: result


@key_callable_results
@clients
def test_key_callable_result_is_trimmed(api, client_cls, result, authorization):
    _call(client_cls(api_key=_key_callable(client_cls, result)), _chat)
    assert _authorizations(api) == [authorization]


class _RecordingHandler(BaseHTTPRequestHandler):
    def __init__(self, authorizations: List[Optional[str]], *args: Any, **kwargs: Any) -> None:
        self.authorizations = authorizations
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        self.authorizations.append(self.headers.get("Authorization"))
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(completion(json.dumps(WEATHER))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        return None


# A real socket, because respx does not run the h11 header validation.
@pytest.fixture
def local_api() -> Iterator[Dict[str, Any]]:
    authorizations: List[Optional[str]] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RecordingHandler, authorizations))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "authorizations": authorizations}
    server.shutdown()
    server.server_close()
    thread.join()


@clients
def test_padded_key_callable_passes_header_validation(local_api, client_cls):
    padded_key = _key_callable(client_cls, f" {CALLABLE_KEY}\n")
    _call(client_cls(api_key=padded_key, base_url=local_api["base_url"]), _chat)
    assert local_api["authorizations"] == [f"Bearer {CALLABLE_KEY}"]


# Each case: how to derive a client from client.openai.
DERIVED = {
    "with_options": lambda openai, **kwargs: openai.with_options(**kwargs),
    "copy": lambda openai, **kwargs: openai.copy(**kwargs),
}
# Each case: the options of the derived client, and the Authorization header its request carries.
DERIVED_KEYS = {
    "keyless": ({"max_retries": 0}, None),
    "new-key": ({"api_key": NEW_KEY}, f"Bearer {NEW_KEY}"),
}


@pytest.mark.parametrize("options,authorization", DERIVED_KEYS.values(), ids=DERIVED_KEYS.keys())
@pytest.mark.parametrize("derive", DERIVED.values(), ids=DERIVED.keys())
@clients
def test_derived_client_from_keyless_client(api, monkeypatch, client_cls, derive, options, authorization):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    _call(derive(client_cls().openai, **options), _list_models)
    assert _authorizations(api) == [authorization]


@clients
def test_keyless_daily_limit_raises_after_one_request(router, client_cls):
    _fail(router, 429, KEYLESS_DAILY_LIMIT, _daily_headers())
    with pytest.raises(RateLimitError) as exc_info:
        _call(client_cls(), _chat)
    assert router.calls.call_count == 1
    assert exc_info.value.code == "keyless_daily_limit_exceeded"
    assert SIGNUP_HINT in str(exc_info.value)


def _keyless_warnings(recorded: List[warnings.WarningMessage]) -> List[warnings.WarningMessage]:
    return [item for item in recorded if str(item.message) == KEYLESS_WARNING]


@clients
def test_first_keyless_client_warns_once(recwarn, client_cls):
    client_cls()
    client_cls()
    (warning,) = _keyless_warnings(recwarn.list)
    assert warning.category is UserWarning
    assert warning.filename == __file__


# Each case: the constructor arguments and the INTERFAZE_API_KEY value of a client that has a key.
KEYED_CLIENTS = {
    "explicit-key": ({"api_key": EXPLICIT_KEY}, ""),
    "env-key": ({}, ENV_KEY),
    "key-callable": ({"api_key": lambda: EXPLICIT_KEY}, ""),
    "authorization-header": ({"default_headers": {"Authorization": f"Bearer {EXPLICIT_KEY}"}}, ""),
    "lowercase-authorization-header": ({"default_headers": {"authorization": f"Bearer {EXPLICIT_KEY}"}}, ""),
}


@pytest.mark.parametrize("kwargs,env_key", KEYED_CLIENTS.values(), ids=KEYED_CLIENTS.keys())
@clients
def test_keyed_client_does_not_warn(recwarn, monkeypatch, client_cls, kwargs, env_key):
    monkeypatch.setenv("INTERFAZE_API_KEY", env_key)
    client_cls(**kwargs)
    assert _keyless_warnings(recwarn.list) == []


@clients
def test_keyed_client_does_not_use_up_the_warning(recwarn, client_cls):
    client_cls(api_key=EXPLICIT_KEY)
    client_cls()
    assert len(_keyless_warnings(recwarn.list)) == 1
