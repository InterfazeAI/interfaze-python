from __future__ import annotations

import asyncio

import httpx
import respx
from conftest import BASIC, CHAT_URL, _chunk, completion, mock_json, mock_sse
from pydantic import BaseModel

from interfaze import AsyncInterfaze, Interfaze

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


def test_escape_hatch_surface_present():
    for comp in (Interfaze(api_key="t").chat.completions, AsyncInterfaze(api_key="t").chat.completions):
        assert hasattr(comp, "parse")
        assert comp.with_raw_response is not None
        assert comp.with_streaming_response is not None
