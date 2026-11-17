from __future__ import annotations

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from . import _inputs as inputs
from ._client import AsyncInterfaze, Interfaze
from ._constants import GUARD_CODES, INTERFAZE_BASE_URL, INTERFAZE_MODEL, TASK_NAMES
from ._errors import InterfazeError
from ._schema import empty_task_schema, response_format
from ._stream import (
    AsyncInterfazeStream,
    InterfazeStream,
    strip_side_channels,
)
from ._stream import (
    _SideChannelFilter as SideChannelFilter,
)
from ._types import (
    GuardCode,
    InterfazeChatCompletion,
    InterfazeParsedChatCompletion,
    Precontext,
    ReasoningEffort,
    TaskName,
)

__version__ = "1.0.4"

__all__ = [
    "Interfaze",
    "AsyncInterfaze",
    "InterfazeError",
    "InterfazeChatCompletion",
    "InterfazeParsedChatCompletion",
    "InterfazeStream",
    "AsyncInterfazeStream",
    "strip_side_channels",
    "SideChannelFilter",
    "Precontext",
    "TaskName",
    "GuardCode",
    "ReasoningEffort",
    "response_format",
    "empty_task_schema",
    "inputs",
    "TASK_NAMES",
    "GUARD_CODES",
    "INTERFAZE_MODEL",
    "INTERFAZE_BASE_URL",
    # openai re-exports
    "OpenAIError",
    "APIError",
    "APIStatusError",
    "APIConnectionError",
    "APITimeoutError",
    "BadRequestError",
    "AuthenticationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "InternalServerError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
]
