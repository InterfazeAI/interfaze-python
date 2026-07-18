from __future__ import annotations

import asyncio
import json

import pytest
import respx
from conftest import JSON_OBJECT, STREAM_CHUNKS, TASK_OCR, TOOL_CALL, last_body, mock_json, mock_sse

from interfaze import AsyncInterfaze, InterfazeError

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


@respx.mock
def test_async_task_and_guard_serialization():
    route = mock_json(TASK_OCR)

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.create(
            task="ocr", guard=["S1", "ALL"], messages=[{"role": "user", "content": "x"}]
        )

    asyncio.run(go())
    system_content = last_body(route)["messages"][0]["content"]
    assert "<task>ocr</task>" in system_content
    assert "<guard>S1, ALL</guard>" in system_content


@respx.mock
def test_async_tool_calls_content_none():
    mock_json(TOOL_CALL)

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.create(
            messages=[{"role": "user", "content": "weather?"}], tools=WEATHER_TOOL, tool_choice="auto"
        )

    r = asyncio.run(go())
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.choices[0].message.content is None
    assert r.choices[0].message.tool_calls[0].function.name == "get_weather"


@respx.mock
def test_async_json_object_fence_stripped():
    mock_json(JSON_OBJECT)

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.create(
            messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
        )

    r = asyncio.run(go())
    content = r.choices[0].message.content
    assert not content.strip().startswith("```")
    assert json.loads(content)["city"] == "Tokyo"


def test_async_missing_key_raises(monkeypatch):
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    with pytest.raises(InterfazeError, match="INTERFAZE_API_KEY"):
        AsyncInterfaze()


@respx.mock
def test_async_create_stream_true_returns_raw_openai_stream():
    mock_sse(STREAM_CHUNKS)

    async def go():
        raw_stream = await AsyncInterfaze(api_key="t").chat.completions.create(
            messages=[{"role": "user", "content": "x"}], stream=True
        )
        return [chunk async for chunk in raw_stream]

    chunks = asyncio.run(go())
    assert len(chunks) == len(STREAM_CHUNKS)
    assert chunks[0].choices[0].delta.content is not None
