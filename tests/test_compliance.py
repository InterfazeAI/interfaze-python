from __future__ import annotations

import asyncio

import httpx
import respx
from conftest import BASIC, CHAT_URL, MIXED_PRECONTEXT, _chunk, completion, mock_json, mock_sse
from pydantic import BaseModel

from interfaze import AsyncInterfaze, Interfaze, InterfazeParsedChatCompletion, Precontext

USAGE_CHUNK = {
    "id": "req-test",
    "object": "chat.completion.chunk",
    "created": 1_700_000_000,
    "model": "interfaze-beta",
    "choices": [],
    "usage": {"prompt_tokens": 216, "completion_tokens": 10, "total_tokens": 226},
    "system_fingerprint": "fp_test",
}
STREAM_WITH_USAGE = [_chunk({"content": "Hi"}), _chunk({}, finish_reason="stop"), USAGE_CHUNK]


@respx.mock
def test_stream_captures_usage_and_fingerprint():
    mock_sse(STREAM_WITH_USAGE)
    stream = Interfaze(api_key="t").chat.completions.stream(
        messages=[{"role": "user", "content": "x"}], stream_options={"include_usage": True}
    )
    final = stream.get_final_completion()
    assert final.usage is not None
    assert final.usage.prompt_tokens == 216
    assert final.usage.completion_tokens == 10
    assert final.usage.total_tokens == 226
    assert final.system_fingerprint == "fp_test"


@respx.mock
def test_async_stream_captures_usage():
    mock_sse(STREAM_WITH_USAGE)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}], stream_options={"include_usage": True}
        )
        async for _ in stream:
            pass
        return await stream.get_final_completion()

    final = asyncio.run(go())
    assert final.usage is not None and final.usage.total_tokens == 226


@respx.mock
def test_request_id_carried_through():
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=BASIC, headers={"x-request-id": "req-abc123"})
    )
    r = Interfaze(api_key="t").chat.completions.create(messages=[{"role": "user", "content": "x"}])
    assert r._request_id == "req-abc123"


class _Weather(BaseModel):
    city: str
    temp_c: float


@respx.mock
def test_parse_returns_parsed_object():
    mock_json(completion('{"city": "Tokyo", "temp_c": 21}'))
    res = Interfaze(api_key="t").chat.completions.parse(
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        response_format=_Weather,
    )
    parsed = res.choices[0].message.parsed
    assert isinstance(parsed, _Weather) and parsed.city == "Tokyo" and parsed.temp_c == 21


_WEATHER_JSON = '{"city": "Tokyo", "temp_c": 21}'
_MSGS = [{"role": "user", "content": "Weather in Tokyo?"}]


@respx.mock
def test_parse_exposes_typed_precontext_and_vcache():
    mock_json(
        completion(
            _WEATHER_JSON,
            precontext=[{"name": "ocr", "result": {"extracted_text": "x"}}],
            vcache=True,
            reasoning="because",
        )
    )
    res = Interfaze(api_key="t").chat.completions.parse(messages=_MSGS, response_format=_Weather)
    assert isinstance(res, InterfazeParsedChatCompletion)
    assert res.precontext is not None
    assert isinstance(res.precontext[0], Precontext)
    assert res.precontext[0].name == "ocr"
    assert res.precontext[0].result == {"extracted_text": "x"}
    assert res.vcache is True
    assert res.reasoning == "because"
    parsed = res.choices[0].message.parsed
    assert isinstance(parsed, _Weather) and parsed.city == "Tokyo"
    assert res.usage is not None and res.usage.total_tokens == 8


@respx.mock
def test_parse_without_precontext_defaults():
    mock_json({k: v for k, v in completion(_WEATHER_JSON).items() if k != "vcache"})
    res = Interfaze(api_key="t").chat.completions.parse(messages=_MSGS, response_format=_Weather)
    assert isinstance(res, InterfazeParsedChatCompletion)
    assert res.precontext is None
    assert res.vcache is False
    parsed = res.choices[0].message.parsed
    assert isinstance(parsed, _Weather) and parsed.temp_c == 21


@respx.mock
def test_parse_keeps_raw_tool_call_precontext_entries():
    mock_json({**MIXED_PRECONTEXT, "choices": completion(_WEATHER_JSON)["choices"]})
    res = Interfaze(api_key="t").chat.completions.parse(messages=_MSGS, response_format=_Weather)
    assert res.precontext is not None and len(res.precontext) == 2
    assert res.precontext[0].result == {"extracted_text": "x"}
    assert res.precontext[1].name is None and res.precontext[1].result is None
    assert res.precontext[1].model_extra == {
        "toolCallId": "call_1",
        "toolName": "run_code",
        "input": {"code": "print(1)"},
    }


@respx.mock
def test_parse_carries_request_id():
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=completion(_WEATHER_JSON), headers={"x-request-id": "req-parse1"}
        )
    )
    res = Interfaze(api_key="t").chat.completions.parse(messages=_MSGS, response_format=_Weather)
    assert res._request_id == "req-parse1"


@respx.mock
def test_async_parse_exposes_typed_precontext_and_vcache():
    mock_json(
        completion(
            _WEATHER_JSON, precontext=[{"name": "ocr", "result": {"extracted_text": "x"}}], vcache=True
        )
    )

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.parse(
            messages=_MSGS, response_format=_Weather
        )

    res = asyncio.run(go())
    assert isinstance(res, InterfazeParsedChatCompletion)
    assert res.precontext is not None and isinstance(res.precontext[0], Precontext)
    assert res.precontext[0].result == {"extracted_text": "x"}
    assert res.vcache is True
    parsed = res.choices[0].message.parsed
    assert isinstance(parsed, _Weather) and parsed.city == "Tokyo"


@respx.mock
def test_async_parse_without_precontext():
    mock_json(completion(_WEATHER_JSON))

    async def go():
        return await AsyncInterfaze(api_key="t").chat.completions.parse(
            messages=_MSGS, response_format=_Weather
        )

    res = asyncio.run(go())
    assert isinstance(res, InterfazeParsedChatCompletion)
    assert res.precontext is None and res.vcache is False
    parsed = res.choices[0].message.parsed
    assert isinstance(parsed, _Weather) and parsed.temp_c == 21


def test_escape_hatch_surface_present():
    for comp in (Interfaze(api_key="t").chat.completions, AsyncInterfaze(api_key="t").chat.completions):
        assert hasattr(comp, "parse")
        assert comp.with_raw_response is not None
        assert comp.with_streaming_response is not None
