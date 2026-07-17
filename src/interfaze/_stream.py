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

    def build(self) -> InterfazeChatCompletion:
        text, reasoning, precontext = strip_side_channels(self.content)
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

    def __init__(self, client: OpenAI, kwargs: Dict[str, Any]) -> None:
        self._client = client
        self._kwargs = kwargs
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

    @property
    def text(self) -> str:
        return strip_side_channels(self._state.content)[0]

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
        return self._state.build()


class AsyncInterfazeStream:
    """Async streaming helper — ``async for`` chunks, then ``await get_final_completion()``."""

    def __init__(self, client: AsyncOpenAI, kwargs: Dict[str, Any]) -> None:
        self._client = client
        self._kwargs = kwargs
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
        return self._state.build()
