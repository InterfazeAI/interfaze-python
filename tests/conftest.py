"""Test fixtures mirroring Interfaze wire responses (observed live).

Kept faithful to the actual shapes: `vcache` always present,
`precontext` a list of {name,result},
task content = {name,result} JSON,
json_object content ```json-fenced,
stream deltas role-less.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import respx

CHAT_URL = "https://api.interfaze.ai/v1/chat/completions"

_USAGE = {
    "prompt_tokens": 5,
    "completion_tokens": 3,
    "total_tokens": 8,
}


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
    "Total: $12.34",
    precontext=[{"name": "ocr", "result": {"extracted_text": "Walmart ... TOTAL 12.34"}}],
)
# A task entry plus a raw tool-call entry (server-appended on tool/run_code turns).
MIXED_PRECONTEXT = completion(
    "Ran the code.",
    precontext=[
        {"name": "ocr", "result": {"extracted_text": "x"}},
        {"toolCallId": "call_1", "toolName": "run_code", "input": {"code": "print(1)"}},
    ],
)
REASONING = completion(
    "The sky is blue because...",
    reasoning="Rayleigh scattering means shorter wavelengths...",
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

TASK_OBJECT_DETECTION = completion(
    '{"name": "object_detection", "result": {"objects": [{"label": "bus", "box": [0, 0, 10, 10]}]}}'
)
TASK_GUI_DETECTION = completion(
    '{"name": "gui_detection", "result": {"elements": [{"label": "button", "box": [1, 2, 3, 4]}]}}'
)
TASK_TRANSCRIBE = completion('{"name": "speech_to_text", "result": {"text": "hello world"}}')
TASK_WEB_SEARCH = completion(
    '{"name": "web_search", "result": {"results": [{"title": "AI agents", "url": "https://news.ycombinator.com"}]}}'
)
TASK_SCRAPE = completion('{"name": "scraper", "result": {"text": "Hacker News"}}')
TASK_TRANSLATE = completion('{"name": "translate", "result": "Bonjour"}')
FORECAST_PRECONTEXT = completion(
    "Here is the forecast.", precontext=[{"name": "forecast", "result": {"forecast": [1, 2, 3]}}]
)
FORECAST_FALLBACK = completion("I couldn't run the forecast tool; here's a manual estimate.")


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

# Tool-call arguments split across chunks, as the wire actually streams them.
STREAM_TOOL_CALL_CHUNKS: List[Dict[str, Any]] = [
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"ci'},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Pa'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ris"}'}}]}, finish_reason="tool_calls"),
]

# A <precontext> block with invalid JSON inside — interfaze must swallow it, not crash.
STREAM_MALFORMED_PRECONTEXT_CHUNKS: List[Dict[str, Any]] = [
    _chunk({"content": "<precontext>[not valid json]</precontext>"}),
    _chunk({"content": "Answer anyway."}, finish_reason="stop"),
]

# A heartbeat/ping chunk with no choices at all, as some SSE proxies emit mid-stream.
STREAM_CHUNK_NO_CHOICES: Dict[str, Any] = {
    "id": "req-test",
    "object": "chat.completion.chunk",
    "created": 1_700_000_000,
    "model": "interfaze-beta",
    "choices": [],
}
STREAM_ROLE_THEN_CONTENT_CHUNKS: List[Dict[str, Any]] = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"content": "Hi there."}, finish_reason="stop"),
]

# Two parallel tool calls in a single delta — one arrives complete, one still has no arguments.
STREAM_PARALLEL_TOOL_CALL_CHUNKS: List[Dict[str, Any]] = [
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {"index": 1, "id": "call_2", "type": "function", "function": {"name": "get_time"}},
            ]
        },
        finish_reason="tool_calls",
    ),
]


def mock_json(body: Dict[str, Any]) -> respx.Route:
    """Route POST /chat/completions -> a JSON completion; returns the route (inspect .calls)."""
    return respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))


def mock_status(status: int, body: Dict[str, Any]) -> respx.Route:
    """Route POST /chat/completions -> an error status with a realistic error body."""
    return respx.post(CHAT_URL).mock(return_value=httpx.Response(status, json=body))


def sse_bytes(chunks: List[Dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def mock_sse(chunks: List[Dict[str, Any]]) -> respx.Route:
    return respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_bytes(chunks)
        )
    )


def error_body(message: str, type_: str, code: str) -> Dict[str, Any]:
    """Shape of a real Interfaze error response body."""
    return {"error": {"message": message, "type": type_, "code": code}}


def last_body(route: respx.Route) -> Dict[str, Any]:
    return json.loads(route.calls.last.request.content)


def last_headers(route: respx.Route):
    return route.calls.last.request.headers
