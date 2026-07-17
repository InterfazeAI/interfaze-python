"""LangChain chat model for Interfaze.

Not imported from ``interfaze/__init__.py`` — the core SDK must not require
``langchain-openai``. Import this module directly: ``from interfaze.langchain import ChatInterfaze``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ._constants import INTERFAZE_BASE_URL, INTERFAZE_MODEL
from ._errors import InterfazeError
from ._stream import strip_side_channels

try:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_openai import ChatOpenAI
    from pydantic import Field, SecretStr
except ImportError as exc:
    raise InterfazeError(
        "ChatInterfaze requires the `langchain` extra. Install it with `pip install interfaze[langchain]`."
    ) from exc

_SIDE_FIELDS = ("precontext", "reasoning", "vcache")


def _extract_side_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull Interfaze's non-standard top-level response fields out of a raw chunk/response dict."""
    return {k: data[k] for k in _SIDE_FIELDS if data.get(k) is not None}


def _apply_side_fields(message: AIMessage, side: Dict[str, Any]) -> None:
    for key, value in side.items():
        message.response_metadata[key] = value
        message.additional_kwargs[key] = value


def _strip_tags(message: AIMessage) -> None:
    """Strip inline ``<think>``/``<precontext>`` tags from message content when present."""
    if not isinstance(message.content, str) or not (
        "<think>" in message.content or "<precontext>" in message.content
    ):
        return
    text, reasoning, precontext = strip_side_channels(message.content)
    if text != message.content:
        message.content = text
    if reasoning:
        message.response_metadata.setdefault("reasoning", reasoning)
        message.additional_kwargs.setdefault("reasoning", reasoning)
    if precontext:
        message.response_metadata.setdefault("precontext", precontext)
        message.additional_kwargs.setdefault("precontext", precontext)


def _convert_video_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Interfaze has no native video content type; it accepts video as a file part."""
    if "url" in block:
        file: Dict[str, Any] = {"file_data": block["url"]}
    elif "base64" in block:
        mime = block.get("mime_type", "video/mp4")
        file = {"file_data": f"data:{mime};base64,{block['base64']}"}
    elif "file_id" in block:
        file = {"file_id": block["file_id"]}
    else:
        raise InterfazeError("Video content block requires one of 'url', 'base64', or 'file_id'.")
    extras = block.get("extras")
    if isinstance(extras, dict) and extras.get("filename"):
        file["filename"] = extras["filename"]
    return {"type": "file", "file": file}


def _rewrite_video_blocks(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    rewritten = [
        _convert_video_block(block) if isinstance(block, dict) and block.get("type") == "video" else block
        for block in content
    ]
    return rewritten if rewritten != content else content


class ChatInterfaze(ChatOpenAI):
    """`ChatOpenAI` pointed at Interfaze, with precontext/reasoning/vcache and video support."""

    precontext: Optional[List[Dict[str, Any]]] = Field(default=None)
    """Pre-computed tool output to feed Interfaze (skips its internal tool run); sent via ``extra_body``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("INTERFAZE_API_KEY")
        if not key:
            raise InterfazeError(
                "Missing API key. Pass ChatInterfaze(api_key=...) or set the INTERFAZE_API_KEY "
                "environment variable."
            )
        super().__init__(
            api_key=SecretStr(key),
            base_url=base_url or INTERFAZE_BASE_URL,
            model=model or INTERFAZE_MODEL,
            **kwargs,
        )

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        patched = [
            m.model_copy(update={"content": _rewrite_video_blocks(m.content)})
            if isinstance(m.content, list)
            else m
            for m in messages
        ]
        payload = super()._get_request_payload(patched, stop=stop, **kwargs)
        if self.precontext is not None:
            extra_body = dict(payload.get("extra_body") or {})
            extra_body.setdefault("precontext", self.precontext)
            payload["extra_body"] = extra_body
        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}, warnings=False
            )
        )
        side = _extract_side_fields(response_dict)
        for generation in result.generations:
            message = generation.message
            if isinstance(message, AIMessage):
                _apply_side_fields(message, side)
                _strip_tags(message)
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: Dict[str, Any],
        default_chunk_class: type,
        base_generation_info: Optional[Dict[str, Any]],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk
        message = generation_chunk.message
        if isinstance(message, AIMessage):
            side = _extract_side_fields(chunk)
            if side:
                _apply_side_fields(message, side)
            _strip_tags(message)
        return generation_chunk


__all__ = ["ChatInterfaze"]
