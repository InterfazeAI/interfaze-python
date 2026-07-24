from __future__ import annotations

import asyncio
import json

import respx
from conftest import STREAM_CHUNKS, STREAM_THINK, _chunk, mock_sse

from interfaze import AsyncInterfaze, Interfaze
from interfaze._errors import InterfazeError

TOOL_CALL_CHUNKS = [
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
    _chunk({}, finish_reason="tool_calls"),
]


def _content_deltas(events) -> str:
    return "".join(e.delta for e in events if e.type == "content.delta")


def _assert_no_side_channel_leak(events) -> None:
    for e in events:
        text = str(e)
        assert "<think>" not in text and "<precontext>" not in text


FENCED_JSON = [
    _chunk({"content": "```json\n"}),
    _chunk({"content": '{"city": "Tokyo"}'}),
    _chunk({"content": "\n```"}),
    _chunk({}, finish_reason="stop"),
]


@respx.mock
def test_stream_events_strip_precontext():
    mock_sse(STREAM_CHUNKS)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)
    _assert_no_side_channel_leak(events)

    joined = _content_deltas(events)
    assert "<precontext>" not in joined
    assert "$12.34" in joined

    done = [e for e in events if e.type == "content.done"]
    assert len(done) == 1
    assert "<precontext>" not in done[0].content

    final = stream.get_final_completion()
    content = final.choices[0].message.content or ""
    assert "<precontext>" not in content and "$12.34" in content
    assert final.precontext and final.precontext[0].name == "ocr"
    assert final.choices[0].finish_reason == "stop"


@respx.mock
def test_stream_events_strip_think():
    mock_sse(STREAM_THINK)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)
    _assert_no_side_channel_leak(events)

    joined = _content_deltas(events)
    assert "<think>" not in joined
    assert "The sky is blue." in joined

    final = stream.get_final_completion()
    assert final.reasoning and "Rayleigh" in final.reasoning
    assert "<think>" not in (final.choices[0].message.content or "")


@respx.mock
def test_stream_get_final_completion_without_iterating():
    mock_sse(STREAM_THINK)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    final = stream.get_final_completion()  # consumes internally, no events ever built
    assert final.reasoning and "Rayleigh" in final.reasoning
    assert "<think>" not in (final.choices[0].message.content or "")


@respx.mock
def test_split_tag_across_chunks_no_leak_no_drop():
    chunks = [
        _chunk({"content": "x<pre"}),
        _chunk({"content": 'context>[{"name":"ocr","result":{}}]</precontext>y'}),
        _chunk({}, finish_reason="stop"),
    ]
    mock_sse(chunks)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)
    _assert_no_side_channel_leak(events)
    assert _content_deltas(events) == "xy"


@respx.mock
def test_eof_with_unresolved_tag_prefix_is_flushed_not_dropped():
    # An unterminated "<thi" prefix at EOF must surface as literal text, not be dropped.
    chunks = [_chunk({"content": "Hello <thi"}), _chunk({}, finish_reason="stop")]
    mock_sse(chunks)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)
    assert _content_deltas(events) == "Hello <thi"

    done = [e for e in events if e.type == "content.done"]
    assert done and done[0].content == "Hello <thi"


@respx.mock
def test_literal_less_than_survives():
    chunks = [_chunk({"content": "a < b"}), _chunk({}, finish_reason="stop")]
    mock_sse(chunks)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)
    assert _content_deltas(events) == "a < b"


@respx.mock
def test_tool_call_stream_events_and_final():
    mock_sse(TOOL_CALL_CHUNKS)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    events = list(stream)

    done = [e for e in events if e.type == "tool_calls.function.arguments.done"]
    assert len(done) == 1
    assert done[0].name == "get_weather"
    assert done[0].arguments == '{"city":"Paris"}'
    assert not any(e.type == "content.done" for e in events)

    final = stream.get_final_completion()
    assert final.choices[0].finish_reason == "tool_calls"
    assert final.choices[0].message.content is None
    tool_calls = final.choices[0].message.tool_calls
    assert tool_calls and tool_calls[0].function.name == "get_weather"  # ty:ignore[unresolved-attribute]
    assert tool_calls[0].function.arguments == '{"city":"Paris"}'  # ty:ignore[unresolved-attribute]


@respx.mock
def test_text_deltas_sync():
    mock_sse(STREAM_CHUNKS)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    text = "".join(stream.text_deltas())
    assert "<precontext>" not in text
    assert "$12.34" in text


@respx.mock
def test_double_consume_raises():
    mock_sse(STREAM_CHUNKS)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    list(stream)
    try:
        list(stream)
        assert False, "expected InterfazeError"
    except InterfazeError:
        pass


@respx.mock
def test_async_stream_events():
    mock_sse(STREAM_CHUNKS)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}]
        )
        events = []
        async for e in stream:
            events.append(e)
        return events, await stream.get_final_completion()

    events, final = asyncio.run(go())
    _assert_no_side_channel_leak(events)
    joined = _content_deltas(events)
    assert "<precontext>" not in joined and "$12.34" in joined
    assert final.precontext and final.precontext[0].name == "ocr"


@respx.mock
def test_async_text_deltas():
    mock_sse(STREAM_CHUNKS)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}]
        )
        pieces = []
        async for piece in stream.text_deltas():
            pieces.append(piece)
        return "".join(pieces)

    assert asyncio.run(go()) == "Total is $12.34"


@respx.mock
def test_stream_json_object_strips_fence():
    mock_sse(FENCED_JSON)
    stream = Interfaze(api_key="t").chat.completions.stream(
        messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
    )
    content = stream.get_final_completion().choices[0].message.content or ""
    assert not content.lstrip().startswith("```")
    assert json.loads(content)["city"] == "Tokyo"


@respx.mock
def test_stream_text_property_strips_fence_for_json_object():
    mock_sse(FENCED_JSON)
    stream = Interfaze(api_key="t").chat.completions.stream(
        messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
    )
    stream.get_final_completion()
    assert not stream.text.lstrip().startswith("```")
    assert json.loads(stream.text)["city"] == "Tokyo"


@respx.mock
def test_stream_without_json_object_keeps_fence():
    mock_sse(FENCED_JSON)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    content = stream.get_final_completion().choices[0].message.content or ""
    assert content.lstrip().startswith("```")


@respx.mock
def test_async_stream_json_object_strips_fence():
    mock_sse(FENCED_JSON)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}], response_format={"type": "json_object"}
        )
        return await stream.get_final_completion()

    content = asyncio.run(go()).choices[0].message.content or ""
    assert not content.lstrip().startswith("```")


@respx.mock
def test_async_double_consume_raises():
    mock_sse(STREAM_CHUNKS)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}]
        )
        async for _ in stream:
            pass
        try:
            async for _ in stream:
                pass
            return False
        except InterfazeError:
            return True

    assert asyncio.run(go())
