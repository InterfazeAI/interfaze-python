from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk

from ._errors import InterfazeError
from ._types import InterfazeChatCompletion


def _tag(tag: str) -> "re.Pattern[str]":
    return re.compile(rf"<{tag}>([\s\S]*?)</{tag}>")


def strip_side_channels(content: str) -> Tuple[str, Optional[str], Optional[List[Dict[str, Any]]]]:
    """Pull ``<think>``/``<precontext>`` blocks out of streamed content; return the rest as text."""
    think_re, pre_re = _tag("think"), _tag("precontext")
    thinks = [m.strip() for m in think_re.findall(content)]
    text = pre_re.sub("", think_re.sub("", content))
    pre: List[Dict[str, Any]] = []
    for block in pre_re.findall(content):
        try:
            parsed = json.loads(block.strip())
            pre.extend(parsed if isinstance(parsed, list) else [parsed])
        except (ValueError, TypeError):
            pass
    return text.strip(), ("\n".join(thinks) if thinks else None), (pre or None)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_json_fence(content: str) -> str:
    """Interfaze wraps ``json_object`` content in a ```json fence; unwrap it."""
    t = content.strip()
    if not t.startswith("```"):
        return content
    return _FENCE.sub("", t).strip()


_SIDE_OPEN = ("<think>", "<precontext>")
_SIDE_CLOSE = {"<think>": "</think>", "<precontext>": "</precontext>"}


def _suffix_prefix_len(s: str, tag: str) -> int:
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s[-k:] == tag[:k]:
            return k
    return 0


class _SideChannelFilter:
    """Strip inline ``<think>``/``<precontext>`` blocks from streamed content, chunk by chunk.

    Buffers a trailing partial that may be a split tag; never withholds text that cannot be a tag.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._close: Optional[str] = None

    def feed(self, text: str) -> str:
        self._buf += text
        out: List[str] = []
        while self._buf:
            if self._close is None:
                lt = self._buf.find("<")
                if lt == -1:
                    out.append(self._buf)
                    self._buf = ""
                    break
                if lt:
                    out.append(self._buf[:lt])
                    self._buf = self._buf[lt:]
                opened = next((t for t in _SIDE_OPEN if self._buf.startswith(t)), None)
                if opened:
                    self._close = _SIDE_CLOSE[opened]
                    self._buf = self._buf[len(opened) :]
                    continue
                if any(t.startswith(self._buf) for t in _SIDE_OPEN):
                    break
                out.append("<")
                self._buf = self._buf[1:]
            else:
                end = self._buf.find(self._close)
                if end == -1:
                    keep = _suffix_prefix_len(self._buf, self._close)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[end + len(self._close) :]
                self._close = None
        return "".join(out)

    def flush(self) -> str:
        if self._close is not None:
            self._buf = ""
            return ""
        rest, self._buf = self._buf, ""
        return rest


class _State:
    def __init__(self) -> None:
        self.content = ""
        self.role: Optional[str] = None
        self.finish: Optional[str] = None
        self.id = ""
        self.model = ""
        self.created = 0
        self.tool_calls: Dict[int, Dict[str, str]] = {}

    def accumulate(self, chunk: ChatCompletionChunk) -> None:
        if not self.id and chunk.id:
            self.id = chunk.id
        if not self.model and chunk.model:
            self.model = chunk.model
        if not self.created and chunk.created:
            self.created = chunk.created
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        delta = choice.delta
        if delta and delta.role:
            self.role = delta.role
        if delta and isinstance(delta.content, str):
            self.content += delta.content
        if choice.finish_reason:
            self.finish = choice.finish_reason
        for tc in (delta.tool_calls or []) if delta else []:
            acc = self.tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                acc["id"] = tc.id
            if tc.function and tc.function.name:
                acc["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                acc["arguments"] += tc.function.arguments

    def build(self, strip_fence: bool = False) -> InterfazeChatCompletion:
        text, reasoning, precontext = strip_side_channels(self.content)
        if strip_fence:
            text = strip_json_fence(text)
        tool_calls = [
            {"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in self.tool_calls.values()
        ]
        message: Dict[str, Any] = {"role": self.role or "assistant", "content": None if tool_calls else text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        data: Dict[str, Any] = {
            "id": self.id or "",
            "object": "chat.completion",
            "created": self.created or 0,
            "model": self.model or "interfaze-beta",
            "choices": [
                {"index": 0, "message": message, "finish_reason": self.finish or "stop", "logprobs": None}
            ],
            "vcache": False,
        }
        if reasoning:
            data["reasoning"] = reasoning
        if precontext:
            data["precontext"] = precontext
        return InterfazeChatCompletion.model_validate(data)


class InterfazeStream:
    """Sync streaming helper — iterate chunks, then ``get_final_completion()``."""

    def __init__(self, client: OpenAI, kwargs: Dict[str, Any], strip_fence: bool = False) -> None:
        self._client = client
        self._kwargs = kwargs
        self._strip_fence = strip_fence
        self._state = _State()
        self._started = False
        self._done = False

    def __enter__(self) -> "InterfazeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self) -> "Iterator[ChatCompletionChunk]":
        if self._started:
            raise InterfazeError("This stream has already been consumed.")
        self._started = True
        for chunk in self._client.chat.completions.create(stream=True, **self._kwargs):
            self._state.accumulate(chunk)
            yield chunk
        self._done = True

    def text_deltas(self) -> "Iterator[str]":
        """Yield visible text only, stripping ``<think>``/``<precontext>`` across chunk boundaries.

        Use this (not raw ``create(stream=True)`` deltas) for live rendering. ``reasoning`` and
        ``precontext`` remain available on ``get_final_completion()``.
        """
        if self._started:
            raise InterfazeError("This stream has already been consumed.")
        self._started = True
        filt = _SideChannelFilter()
        for chunk in self._client.chat.completions.create(stream=True, **self._kwargs):
            self._state.accumulate(chunk)
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and isinstance(delta.content, str) and delta.content:
                    visible = filt.feed(delta.content)
                    if visible:
                        yield visible
        tail = filt.flush()
        if tail:
            yield tail
        self._done = True

    @property
    def text(self) -> str:
        text = strip_side_channels(self._state.content)[0]
        return strip_json_fence(text) if self._strip_fence else text

    def get_final_completion(self) -> InterfazeChatCompletion:
        if not self._started:
            self._started = True
            for chunk in self._client.chat.completions.create(stream=True, **self._kwargs):
                self._state.accumulate(chunk)
            self._done = True
        elif not self._done:
            raise InterfazeError(
                "Call get_final_completion() after fully iterating, or instead of iterating."
            )
        return self._state.build(self._strip_fence)


class AsyncInterfazeStream:
    """Async streaming helper — ``async for`` chunks, then ``await get_final_completion()``."""

    def __init__(self, client: AsyncOpenAI, kwargs: Dict[str, Any], strip_fence: bool = False) -> None:
        self._client = client
        self._kwargs = kwargs
        self._strip_fence = strip_fence
        self._state = _State()
        self._started = False
        self._done = False

    async def __aenter__(self) -> "AsyncInterfazeStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def __aiter__(self) -> "AsyncIterator[ChatCompletionChunk]":
        if self._started:
            raise InterfazeError("This stream has already been consumed.")
        self._started = True
        stream = await self._client.chat.completions.create(stream=True, **self._kwargs)
        async for chunk in stream:
            self._state.accumulate(chunk)
            yield chunk
        self._done = True

    async def text_deltas(self) -> "AsyncIterator[str]":
        """Yield visible text only, stripping ``<think>``/``<precontext>`` across chunk boundaries.

        Use this (not raw ``create(stream=True)`` deltas) for live rendering. ``reasoning`` and
        ``precontext`` remain available on ``get_final_completion()``.
        """
        if self._started:
            raise InterfazeError("This stream has already been consumed.")
        self._started = True
        filt = _SideChannelFilter()
        stream = await self._client.chat.completions.create(stream=True, **self._kwargs)
        async for chunk in stream:
            self._state.accumulate(chunk)
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and isinstance(delta.content, str) and delta.content:
                    visible = filt.feed(delta.content)
                    if visible:
                        yield visible
        tail = filt.flush()
        if tail:
            yield tail
        self._done = True

    async def get_final_completion(self) -> InterfazeChatCompletion:
        if not self._started:
            self._started = True
            stream = await self._client.chat.completions.create(stream=True, **self._kwargs)
            async for chunk in stream:
                self._state.accumulate(chunk)
            self._done = True
        elif not self._done:
            raise InterfazeError(
                "Call get_final_completion() after fully iterating, or instead of iterating."
            )
        return self._state.build(self._strip_fence)
