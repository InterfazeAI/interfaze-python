from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_openai")

import respx
from assets import ASSETS
from conftest import BASIC, STREAM_CHUNKS, _chunk, completion, last_body, mock_json, mock_sse

from interfaze import InterfazeError
from interfaze._constants import INTERFAZE_BASE_URL, INTERFAZE_MODEL
from interfaze.langchain import ChatInterfaze
from langchain_core.messages import HumanMessage

CUSTOM_FIELDS = completion(
    "Hello there",
    precontext=[{"name": "ocr", "result": {"extracted_text": "x"}}],
    reasoning="because reasons",
    vcache=True,
)


# ---- defaults ----
def test_defaults_point_at_interfaze():
    model = ChatInterfaze(api_key="t")
    assert model.openai_api_base == INTERFAZE_BASE_URL
    assert model.model_name == INTERFAZE_MODEL


def test_defaults_overridable():
    model = ChatInterfaze(api_key="t", base_url="https://example.com/v1", model="other-model")
    assert model.openai_api_base == "https://example.com/v1"
    assert model.model_name == "other-model"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    with pytest.raises(InterfazeError, match="Missing API key"):
        ChatInterfaze()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("INTERFAZE_API_KEY", "env-key")
    model = ChatInterfaze()
    assert model.openai_api_key is not None


# ---- custom response fields ----
@respx.mock
def test_custom_response_fields_surfaced():
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert result.response_metadata["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.response_metadata["reasoning"] == "because reasons"
    assert result.response_metadata["vcache"] is True
    assert result.additional_kwargs["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.additional_kwargs["reasoning"] == "because reasons"
    assert result.additional_kwargs["vcache"] is True


@respx.mock
def test_response_without_precontext_or_reasoning_unaffected():
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert "precontext" not in result.response_metadata
    assert "reasoning" not in result.response_metadata
    assert result.response_metadata["vcache"] is False
    assert result.content == "Hi!"


# ---- request-side precontext ----
@respx.mock
def test_request_precontext_injected():
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t", precontext=[{"name": "ocr", "result": {"extracted_text": "y"}}])
    model.invoke([HumanMessage("hi")])
    body = last_body(route)
    assert body["precontext"] == [{"name": "ocr", "result": {"extracted_text": "y"}}]


@respx.mock
def test_request_without_precontext_field_omits_it():
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke([HumanMessage("hi")])
    body = last_body(route)
    assert "precontext" not in body


# ---- video content blocks ----
@respx.mock
def test_video_block_converted_to_file_part():
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "what happens in this clip?"},
            {"type": "video", "url": ASSETS["video"]},
        ]
    )
    model.invoke([message])  # must not raise
    body = last_body(route)
    content = body["messages"][-1]["content"]
    assert {"type": "file", "file": {"file_data": ASSETS["video"]}} in content


@respx.mock
def test_video_block_base64_converted_to_file_part():
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    message = HumanMessage(content=[{"type": "video", "base64": "AAAA", "mime_type": "video/mp4"}])
    model.invoke([message])
    body = last_body(route)
    content = body["messages"][-1]["content"]
    assert content[0]["type"] == "file"
    assert content[0]["file"]["file_data"] == "data:video/mp4;base64,AAAA"


# ---- inline tag stripping (streaming) ----
@respx.mock
def test_streaming_strips_inline_tags_and_carries_precontext():
    mock_sse(STREAM_CHUNKS)
    model = ChatInterfaze(api_key="t")
    chunks = list(model.stream([HumanMessage("x")]))
    text = "".join(c.content for c in chunks)  # ty:ignore[no-matching-overload]
    assert "<precontext>" not in text
    assert text == "Total is $12.34"
    precontext_chunks = [c for c in chunks if c.additional_kwargs.get("precontext")]
    assert precontext_chunks
    assert precontext_chunks[0].additional_kwargs["precontext"][0]["name"] == "ocr"


THINK_SPLIT = [
    _chunk({"content": "<th"}),
    _chunk({"content": "ink>Rayleigh scat"}),
    _chunk({"content": "tering.</think>The sky "}),
    _chunk({"content": "is blue."}),
    _chunk({}, finish_reason="stop"),
]


@respx.mock
def test_streaming_recovers_reasoning_split_across_chunks():
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    chunks = list(model.stream([HumanMessage("x")]))
    text = "".join(c.content for c in chunks)  # ty:ignore[no-matching-overload]
    assert "<think>" not in text and text == "The sky is blue."
    reasoning = [c for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning and reasoning[0].additional_kwargs["reasoning"] == "Rayleigh scattering."


@respx.mock
def test_async_streaming_recovers_reasoning_split_across_chunks():
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")

    async def go():
        return [c async for c in model.astream([HumanMessage("x")])]

    chunks = asyncio.run(go())
    text = "".join(c.content for c in chunks)
    assert "<think>" not in text and text == "The sky is blue."
    reasoning = [c for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning and reasoning[0].additional_kwargs["reasoning"] == "Rayleigh scattering."


# ---- async ----
@respx.mock
def test_async_invoke_surfaces_side_fields():
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")

    async def go():
        return await model.ainvoke([HumanMessage("hi")])

    result = asyncio.run(go())
    assert result.response_metadata["precontext"][0]["name"] == "ocr"
    assert result.response_metadata["vcache"] is True
