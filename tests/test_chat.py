from __future__ import annotations

import asyncio
import json

import pytest
import respx
from conftest import (
    BASIC,
    JSON_OBJECT,
    MIXED_PRECONTEXT,
    PRECONTEXT,
    REASONING,
    STREAM_CHUNKS,
    TASK_OCR,
    TOOL_CALL,
    completion,
    last_body,
    last_headers,
    mock_json,
    mock_sse,
)

from interfaze import AsyncInterfaze, Interfaze, InterfazeChatCompletion, InterfazeError
from interfaze._chat import to_interfaze

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


# ---- request serialization ----
@respx.mock
def test_base_url_and_model_default():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert route.calls.last.request.url == "https://api.interfaze.ai/v1/chat/completions"
    assert last_body(route)["model"] == "interfaze-beta"


@respx.mock
def test_task_serialization():
    route = mock_json(TASK_OCR)
    Interfaze(api_key="t").chat.completions.create(task="ocr", messages=[{"role": "user", "content": "x"}])
    body = last_body(route)
    assert body["messages"][0]["role"] == "system" and "<task>ocr</task>" in body["messages"][0]["content"]
    assert body["response_format"]["json_schema"]["schema"] == {}


@respx.mock
def test_guard_serialization():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(
        guard=["S1", "S12_IMAGE"], messages=[{"role": "user", "content": "x"}]
    )
    assert "<guard>S1, S12_IMAGE</guard>" in last_body(route)["messages"][0]["content"]


@respx.mock
def test_forecast_is_a_valid_task():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(
        task="forecast", messages=[{"role": "user", "content": "x"}]
    )
    assert "<task>forecast</task>" in last_body(route)["messages"][0]["content"]


@respx.mock
def test_tags_merge_into_existing_system_message():
    route = mock_json(TASK_OCR)
    Interfaze(api_key="t").chat.completions.create(
        task="ocr",
        guard=["S1"],
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "x"},
        ],
    )
    systems = [m for m in last_body(route)["messages"] if m["role"] == "system"]
    assert len(systems) == 1
    assert "<task>ocr</task>" in systems[0]["content"]
    assert "<guard>S1</guard>" in systems[0]["content"]
    assert "You are helpful." in systems[0]["content"]


def test_task_plus_nonempty_schema_raises():
    with pytest.raises(InterfazeError, match="non-empty"):
        Interfaze(api_key="t").chat.completions.create(
            task="ocr",
            messages=[{"role": "user", "content": "x"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "s",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                },
            },
        )


@respx.mock
def test_control_headers():
    route = mock_json(BASIC)
    Interfaze(api_key="t", show_additional_info=True, bypass_cache=True).chat.completions.create(
        messages=[{"role": "user", "content": "x"}]
    )
    h = last_headers(route)
    assert h["x-show-additional-info"] == "true" and h["x-bypass-cache"] == "true"


@respx.mock
def test_all_control_headers_present():
    route = mock_json(BASIC)
    Interfaze(
        api_key="t",
        show_additional_info=True,
        bypass_moe=True,
        bypass_cache=True,
        admin_key="admin-secret",
    ).chat.completions.create(messages=[{"role": "user", "content": "x"}])
    h = last_headers(route)
    assert h["x-show-additional-info"] == "true"
    assert h["x-bypass-moe"] == "true"
    assert h["x-bypass-cache"] == "true"
    assert h["x-admin-key"] == "admin-secret"


@respx.mock
def test_default_headers_omit_control_flags_when_unset():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "x"}])
    h = last_headers(route)
    assert "x-show-additional-info" not in h
    assert "x-bypass-moe" not in h
    assert "x-bypass-cache" not in h
    assert "x-admin-key" not in h


@respx.mock
def test_per_request_extra_headers_override_client_default():
    route = mock_json(BASIC)
    Interfaze(api_key="t", admin_key="client-default").chat.completions.create(
        messages=[{"role": "user", "content": "x"}], extra_headers={"x-admin-key": "per-request-override"}
    )
    assert last_headers(route)["x-admin-key"] == "per-request-override"


@respx.mock
def test_reasoning_effort_on_and_extra_body_forwarded():
    route = mock_json(BASIC)
    Interfaze(api_key="t").chat.completions.create(
        reasoning_effort="on", messages=[{"role": "user", "content": "x"}], extra_body={"custom": True}
    )
    body = last_body(route)
    assert body["reasoning_effort"] == "on" and body["custom"] is True


# ---- response mapping ----
@respx.mock
def test_precontext_and_vcache_typed():
    mock_json(PRECONTEXT)
    r = Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "x"}])
    assert isinstance(r, InterfazeChatCompletion)
    assert r.precontext and r.precontext[0].name == "ocr"
    assert isinstance(r.vcache, bool)


@respx.mock
def test_precontext_tolerates_raw_toolcall_entries():
    # raw tool-call entries in precontext must not raise.
    mock_json(MIXED_PRECONTEXT)
    r = Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "run code"}])
    assert isinstance(r, InterfazeChatCompletion)
    assert r.precontext is not None
    assert len(r.precontext) == 2
    assert r.precontext[0].name == "ocr" and r.precontext[0].result == {"extracted_text": "x"}
    assert r.precontext[1].name is None
    assert (r.precontext[1].model_extra or {}).get("toolName") == "run_code"


@respx.mock
def test_reasoning_typed():
    mock_json(REASONING)
    r = Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "x"}])
    assert r.reasoning and "Rayleigh" in r.reasoning


@respx.mock
def test_json_object_fence_stripped():
    mock_json(JSON_OBJECT)
    r = Interfaze(api_key="t").chat.completions.create(
        messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
    )
    content = r.choices[0].message.content
    assert content is not None
    assert not content.strip().startswith("```")
    assert json.loads(content)["city"] == "Tokyo"


@respx.mock
def test_tools_content_none():
    mock_json(TOOL_CALL)
    r = Interfaze(api_key="t").chat.completions.create(
        messages=[{"role": "user", "content": "weather?"}], tools=WEATHER_TOOL, tool_choice="auto"
    )
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.choices[0].message.content is None
    assert r.choices[0].message.tool_calls[0].function.name == "get_weather"


@respx.mock
def test_usage_tokens_surfaced():
    mock_json(BASIC)
    r = Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert r.usage is not None
    assert r.usage.prompt_tokens == 5
    assert r.usage.completion_tokens == 3
    assert r.usage.total_tokens == 8


@respx.mock
def test_json_object_requested_but_content_not_fenced_passthrough():
    mock_json(completion('{"city": "Tokyo"}'))
    r = Interfaze(api_key="t").chat.completions.create(
        messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
    )
    assert json.loads(r.choices[0].message.content)["city"] == "Tokyo"


@respx.mock
def test_json_object_with_tool_call_content_none_no_crash():
    """A json_object response_format combined with a tool-call response (content=None) must
    not crash the fence-stripping pass — it only strips string content."""
    mock_json(TOOL_CALL)
    r = Interfaze(api_key="t").chat.completions.create(
        messages=[{"role": "user", "content": "weather?"}],
        tools=WEATHER_TOOL,
        response_format={"type": "json_object"},
    )
    assert r.choices[0].message.content is None
    assert r.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_to_interfaze_tolerates_missing_choices():
    """Defensive parsing: a malformed/edge-case completion with no choices must not crash
    fence-stripping — the surrounding try/except in `to_interfaze` swallows the IndexError."""

    class FakeRaw:
        def model_dump(self):
            return {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "vcache": False,
            }

    result = to_interfaze(FakeRaw(), strip_fence=True)
    assert result.choices == []


@respx.mock
def test_create_stream_true_returns_raw_openai_stream():
    """`create(stream=True)` is the low-level escape hatch — it hands back openai's own
    Stream of raw chunks, not the InterfazeStream wrapper `.stream()` returns."""
    mock_sse(STREAM_CHUNKS)
    raw_stream = Interfaze(api_key="t").chat.completions.create(
        messages=[{"role": "user", "content": "x"}], stream=True
    )
    chunks = list(raw_stream)
    assert len(chunks) == len(STREAM_CHUNKS)
    assert chunks[0].choices[0].delta.content is not None


# ---- async ----
@respx.mock
def test_async_mapping():
    mock_json(PRECONTEXT)

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.create(
            messages=[{"role": "user", "content": "x"}]
        )

    r = asyncio.run(go())
    assert isinstance(r, InterfazeChatCompletion) and r.precontext[0].name == "ocr"
