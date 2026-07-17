from __future__ import annotations

import asyncio

import respx
from conftest import STREAM_CHUNKS, STREAM_THINK, mock_sse

from interfaze import AsyncInterfaze, Interfaze


@respx.mock
def test_stream_iterates_and_accumulates():
    mock_sse(STREAM_CHUNKS)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    chunks = list(stream)
    assert len(chunks) == len(STREAM_CHUNKS)
    final = stream.get_final_completion()
    content = final.choices[0].message.content or ""
    assert "<precontext>" not in content and "$12.34" in content
    assert final.precontext and final.precontext[0].name == "ocr"
    assert final.choices[0].finish_reason == "stop"


@respx.mock
def test_stream_think_parsed_without_iterating():
    mock_sse(STREAM_THINK)
    stream = Interfaze(api_key="t").chat.completions.stream(messages=[{"role": "user", "content": "x"}])
    final = stream.get_final_completion()  # consumes internally
    assert final.reasoning and "Rayleigh" in final.reasoning
    assert "<think>" not in (final.choices[0].message.content or "")


@respx.mock
def test_async_stream():
    mock_sse(STREAM_CHUNKS)

    async def go():
        stream = AsyncInterfaze(api_key="t").chat.completions.stream(
            messages=[{"role": "user", "content": "x"}]
        )
        n = 0
        async for _ in stream:
            n += 1
        return n, await stream.get_final_completion()

    n, final = asyncio.run(go())
    assert n == len(STREAM_CHUNKS)
    assert final.precontext and final.precontext[0].name == "ocr"
