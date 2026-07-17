"""Test fixtures mirroring REAL Interfaze wire responses (observed live).

Kept faithful to the actual shapes: `vcache` always present, `precontext` a list of
{name,result}, task content = {name,result} JSON, json_object content ```json-fenced,
stream deltas role-less.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import respx

CHAT_URL = "https://api.interfaze.ai/v1/chat/completions"

_USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def completion(
    content: Any = "Hi!", *, finish_reason: str = "stop", tool_calls=None, **extra: Any
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        message["content"] = None
    body: Dict[str, Any] = {
        "id": "req-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "interfaze-beta",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "usage": _USAGE,
        "vcache": False,
    }
    body.update(extra)
    return body


BASIC = completion("Hi!")
PRECONTEXT = completion(
    "Total: $12.34", precontext=[{"name": "ocr", "result": {"extracted_text": "Walmart ... TOTAL 12.34"}}]
)
# Mixed precontext: a well-formed task entry + a RAW model tool-call entry (as the server
# appends on any tool / run_code turn) — {toolCallId, toolName, input}, no name/result.
MIXED_PRECONTEXT = completion(
    "Ran the code.",
    precontext=[
        {"name": "ocr", "result": {"extracted_text": "x"}},
        {"toolCallId": "call_1", "toolName": "run_code", "input": {"code": "print(1)"}},
    ],
)
REASONING = completion(
    "The sky is blue because...", reasoning="Rayleigh scattering means shorter wavelengths..."
)
JSON_OBJECT = completion('```json\n{\n  "city": "Tokyo",\n  "temp_c": 21\n}\n```')
TASK_OCR = completion('{"name": "ocr", "result": {"extracted_text": "See back of receipt", "width": 800}}')
TOOL_CALL = completion(
    None,
    finish_reason="tool_calls",
    tool_calls=[
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ],
)


def _chunk(delta: Dict[str, Any], finish_reason=None) -> Dict[str, Any]:
    return {
        "id": "req-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "interfaze-beta",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# role-less deltas + a <precontext> block (as Interfaze streams them)
STREAM_CHUNKS: List[Dict[str, Any]] = [
    _chunk({"content": '<precontext>[{"name":"ocr","result":{"extracted_text":"x"}}]</precontext>'}),
    _chunk({"content": "Total "}),
    _chunk({"content": "is $12.34"}),
    _chunk({}, finish_reason="stop"),
]
STREAM_THINK: List[Dict[str, Any]] = [
    _chunk({"content": "<think>Rayleigh scattering.</think>"}),
    _chunk({"content": "The sky is blue."}),
    _chunk({}, finish_reason="stop"),
]


def mock_json(body: Dict[str, Any]) -> respx.Route:
    """Route POST /chat/completions -> a JSON completion; returns the route (inspect .calls)."""
    return respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))


def sse_bytes(chunks: List[Dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def mock_sse(chunks: List[Dict[str, Any]]) -> respx.Route:
    return respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_bytes(chunks)
        )
    )


def last_body(route: respx.Route) -> Dict[str, Any]:
    return json.loads(route.calls.last.request.content)


def last_headers(route: respx.Route):
    return route.calls.last.request.headers
