from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast, overload

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion

from ._constants import INTERFAZE_MODEL
from ._errors import InterfazeError
from ._guard import guard_tag
from ._schema import empty_task_schema
from ._stream import AsyncInterfazeStream, InterfazeStream, strip_json_fence
from ._types import GuardCode, InterfazeChatCompletion, TaskName


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
    msgs = list(messages)
    if tags:
        idx = next(
            (i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        if idx is not None and isinstance(msgs[idx].get("content"), str):
            merged = dict(msgs[idx])
            existing = merged["content"]
            merged["content"] = f"{tags}\n{existing}" if existing else tags
            msgs[idx] = merged
        else:
            msgs = [{"role": "system", "content": tags}, *msgs]
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
    result = InterfazeChatCompletion.model_validate(data)
    request_id = getattr(raw, "_request_id", None)
    if request_id is not None:
        result._request_id = request_id
    return result


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
        kw, strip = self._kwargs(messages, model, task, guard, response_format, kwargs)
        return InterfazeStream(self._client, kw, strip)

    def parse(
        self,
        *,
        messages: Iterable[Any],
        response_format: Any,
        model: str = INTERFAZE_MODEL,
        guard: Optional[List[GuardCode]] = None,
        **kwargs: Any,
    ) -> "ParsedChatCompletion[Any]":
        """Structured output via a Pydantic model (delegates to the OpenAI client's ``parse``).

        Interfaze extras (``vcache``/``precontext``) are present but untyped here; use ``create`` for
        the typed extended completion.
        """
        msgs = prepare(messages, model, None, guard, None)[0]
        return self._client.chat.completions.parse(
            model=model, messages=msgs, response_format=response_format, **kwargs
        )

    @property
    def with_raw_response(self) -> Any:
        """Raw-HTTP escape hatch (delegates to the OpenAI client; bypasses task/guard preprocessing)."""
        return self._client.chat.completions.with_raw_response

    @property
    def with_streaming_response(self) -> Any:
        """Streaming raw-HTTP escape hatch (delegates to the OpenAI client)."""
        return self._client.chat.completions.with_streaming_response


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
        kw, strip = self._kwargs(messages, model, task, guard, response_format, kwargs)
        return AsyncInterfazeStream(self._client, kw, strip)

    async def parse(
        self,
        *,
        messages: Iterable[Any],
        response_format: Any,
        model: str = INTERFAZE_MODEL,
        guard: Optional[List[GuardCode]] = None,
        **kwargs: Any,
    ) -> "ParsedChatCompletion[Any]":
        """Structured output via a Pydantic model (delegates to the OpenAI client's ``parse``).

        Interfaze extras (``vcache``/``precontext``) are present but untyped here; use ``create`` for
        the typed extended completion.
        """
        msgs = prepare(messages, model, None, guard, None)[0]
        return await self._client.chat.completions.parse(
            model=model, messages=msgs, response_format=response_format, **kwargs
        )

    @property
    def with_raw_response(self) -> Any:
        """Raw-HTTP escape hatch (delegates to the OpenAI client; bypasses task/guard preprocessing)."""
        return self._client.chat.completions.with_raw_response

    @property
    def with_streaming_response(self) -> Any:
        """Streaming raw-HTTP escape hatch (delegates to the OpenAI client)."""
        return self._client.chat.completions.with_streaming_response


class Chat:
    def __init__(self, client: OpenAI) -> None:
        self.completions = Completions(client)


class AsyncChat:
    def __init__(self, client: AsyncOpenAI) -> None:
        self.completions = AsyncCompletions(client)
