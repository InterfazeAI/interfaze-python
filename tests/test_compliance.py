from __future__ import annotations

import asyncio

import httpx
import respx
from conftest import BASIC, CHAT_URL, _chunk, mock_sse

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
