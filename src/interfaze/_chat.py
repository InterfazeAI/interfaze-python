from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast, overload

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ._constants import INTERFAZE_MODEL
from ._errors import InterfazeError
from ._guard import guard_tag
from ._schema import empty_task_schema
from ._stream import AsyncInterfazeStream, InterfazeStream
from ._types import GuardCode, InterfazeChatCompletion, TaskName

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_json_fence(content: str) -> str:
    """Interfaze wraps ``json_object`` content in a ```json fence; unwrap it."""
    t = content.strip()
    if not t.startswith("```"):
        return content
    return _FENCE.sub("", t).strip()


def _is_non_empty_schema(rf: Any) -> bool:
    schema = (rf or {}).get("json_schema", {}).get("schema", {}) if isinstance(rf, dict) else {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    return bool(props)


def prepare(
    messages: Iterable[Any],
    model: Optional[str],
    task: Optional[TaskName],
    guard: Optional[List[GuardCode]],
    response_format: Optional[Dict[str, Any]],
) -> "tuple[List[Any], str, Optional[Dict[str, Any]], bool]":
    rf = response_format
    if task:
        if rf and _is_non_empty_schema(rf):
            raise InterfazeError(
                "A non-empty `response_format` cannot be combined with `task` "
                "(Interfaze runs tasks with raw output)."
            )
        rf = empty_task_schema()
    tags = " ".join(
        t for t in (f"<task>{task}</task>" if task else None, guard_tag(guard) if guard else None) if t
    )
    msgs = [{"role": "system", "content": tags}, *messages] if tags else list(messages)
    strip = isinstance(rf, dict) and rf.get("type") == "json_object"
    return msgs, (model or INTERFAZE_MODEL), rf, strip


def to_interfaze(raw: ChatCompletion, strip_fence: bool) -> InterfazeChatCompletion:
    data = raw.model_dump()
    if strip_fence:
        try:
            msg = data["choices"][0]["message"]
            if isinstance(msg.get("content"), str):
                msg["content"] = strip_json_fence(msg["content"])
        except (KeyError, IndexError, TypeError):
            pass
    return InterfazeChatCompletion.model_validate(data)


class _CompletionsBase:
    def _kwargs(
        self,
        messages: Iterable[Any],
        model: Optional[str],
        task: Optional[TaskName],
        guard: Optional[List[GuardCode]],
        response_format: Optional[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> "tuple[Dict[str, Any], bool]":
        msgs, mdl, rf, strip = prepare(messages, model, task, guard, response_format)
        kw: Dict[str, Any] = {"model": mdl, "messages": msgs, **extra}
        if rf is not None:
            kw["response_format"] = rf
        return kw, strip


class Completions(_CompletionsBase):
    """`interfaze.chat.completions` (sync)."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        messages: Iterable[Any],
        stream: Literal[False] = False,
        model: str = ...,
        task: Optional[TaskName] = ...,
        guard: Optional[List[GuardCode]] = ...,
        response_format: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> InterfazeChatCompletion: ...
    @overload
    def create(
        self,
        *,
        messages: Iterable[Any],
        stream: Literal[True],
        model: str = ...,
        task: Optional[TaskName] = ...,
        guard: Optional[List[GuardCode]] = ...,
        response_format: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> Stream[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        messages: Iterable[Any],
        stream: bool = False,
        model: str = INTERFAZE_MODEL,
        task: Optional[TaskName] = None,
        guard: Optional[List[GuardCode]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Union[InterfazeChatCompletion, Stream[ChatCompletionChunk]]":
        kw, strip = self._kwargs(messages, model, task, guard, response_format, kwargs)
        if stream:
            return cast(
                "Stream[ChatCompletionChunk]", self._client.chat.completions.create(stream=True, **kw)
            )
        raw = self._client.chat.completions.create(**kw)
        return to_interfaze(cast(ChatCompletion, raw), strip)

    def stream(
        self,
        *,
        messages: Iterable[Any],
        model: str = INTERFAZE_MODEL,
        task: Optional[TaskName] = None,
        guard: Optional[List[GuardCode]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> InterfazeStream:
        kw, _ = self._kwargs(messages, model, task, guard, response_format, kwargs)
        return InterfazeStream(self._client, kw)


class AsyncCompletions(_CompletionsBase):
    """`interfaze.chat.completions` (async)."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        messages: Iterable[Any],
        stream: Literal[False] = False,
        model: str = ...,
        task: Optional[TaskName] = ...,
        guard: Optional[List[GuardCode]] = ...,
        response_format: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> InterfazeChatCompletion: ...
    @overload
    async def create(
        self,
        *,
        messages: Iterable[Any],
        stream: Literal[True],
        model: str = ...,
        task: Optional[TaskName] = ...,
        guard: Optional[List[GuardCode]] = ...,
        response_format: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    async def create(
        self,
        *,
        messages: Iterable[Any],
        stream: bool = False,
        model: str = INTERFAZE_MODEL,
        task: Optional[TaskName] = None,
        guard: Optional[List[GuardCode]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Union[InterfazeChatCompletion, AsyncStream[ChatCompletionChunk]]":
        kw, strip = self._kwargs(messages, model, task, guard, response_format, kwargs)
        if stream:
            return cast(
                "AsyncStream[ChatCompletionChunk]",
                await self._client.chat.completions.create(stream=True, **kw),
            )
        raw = await self._client.chat.completions.create(**kw)
        return to_interfaze(cast(ChatCompletion, raw), strip)

    def stream(
        self,
        *,
        messages: Iterable[Any],
        model: str = INTERFAZE_MODEL,
        task: Optional[TaskName] = None,
        guard: Optional[List[GuardCode]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncInterfazeStream:
        kw, _ = self._kwargs(messages, model, task, guard, response_format, kwargs)
        return AsyncInterfazeStream(self._client, kw)


class Chat:
    def __init__(self, client: OpenAI) -> None:
        self.completions = Completions(client)


class AsyncChat:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.completions = AsyncCompletions(client)
