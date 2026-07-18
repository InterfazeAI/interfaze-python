from __future__ import annotations

import asyncio

import pytest
import respx
from assets import ASSETS
from conftest import (
    FORECAST_FALLBACK,
    FORECAST_PRECONTEXT,
    TASK_GUI_DETECTION,
    TASK_OBJECT_DETECTION,
    TASK_OCR,
    TASK_SCRAPE,
    TASK_TRANSCRIBE,
    TASK_TRANSLATE,
    TASK_WEB_SEARCH,
    completion,
    last_body,
    mock_json,
)

from interfaze import AsyncInterfaze, Interfaze
from interfaze._tasks import _extract

IS_ASYNC = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


def test_extract_falsy_content_returns_none():
    assert _extract(None) is None
    assert _extract("") is None


def test_extract_dict_without_result_key_returns_whole_dict():
    assert _extract('{"name": "ocr"}') == {"name": "ocr"}


def test_extract_non_dict_json_returns_as_is():
    assert _extract("[1, 2, 3]") == [1, 2, 3]


def test_extract_non_json_string_returns_raw_content():
    assert _extract("plain text result, not JSON") == "plain text result, not JSON"


def call_task(is_async: bool, method: str, *args, **kwargs):
    client = AsyncInterfaze(api_key="t") if is_async else Interfaze(api_key="t")
    fn = getattr(client.tasks, method)
    return asyncio.run(fn(*args, **kwargs)) if is_async else fn(*args, **kwargs)


@IS_ASYNC
@respx.mock
def test_ocr(is_async):
    route = mock_json(TASK_OCR)
    result = call_task(is_async, "ocr", ASSETS["image"])
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert system_msg["role"] == "system" and "<task>ocr</task>" in system_msg["content"]
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "Extract all text and data."}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"] == ASSETS["image"]
    assert result == {"extracted_text": "See back of receipt", "width": 800}


@IS_ASYNC
@respx.mock
def test_object_detection(is_async):
    route = mock_json(TASK_OBJECT_DETECTION)
    result = call_task(is_async, "object_detection", ASSETS["scene"])
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>object_detection</task>" in system_msg["content"]
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "Detect all objects."}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"] == ASSETS["scene"]
    assert result == {"objects": [{"label": "bus", "box": [0, 0, 10, 10]}]}


@IS_ASYNC
@respx.mock
def test_gui_detection(is_async):
    """ASSETS["gui"] has no file extension, so auto_part falls through to a generic `file` part
    (not `image_url`) — the gui_detection helper has no way to hint the mime type."""
    route = mock_json(TASK_GUI_DETECTION)
    result = call_task(is_async, "gui_detection", ASSETS["gui"])
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>gui_detection</task>" in system_msg["content"]
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "Detect all GUI elements."}
    assert parts[1]["type"] == "file" and parts[1]["file"]["file_data"] == ASSETS["gui"]
    assert result == {"elements": [{"label": "button", "box": [1, 2, 3, 4]}]}


@IS_ASYNC
@respx.mock
def test_transcribe(is_async):
    route = mock_json(TASK_TRANSCRIBE)
    result = call_task(is_async, "transcribe", ASSETS["audio"])
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>speech_to_text</task>" in system_msg["content"]
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "Transcribe this audio."}
    assert parts[1]["type"] == "input_audio" and parts[1]["input_audio"]["data"] == ASSETS["audio"]
    assert parts[1]["input_audio"]["format"] == "wav"
    assert result == {"text": "hello world"}


@IS_ASYNC
@respx.mock
def test_web_search(is_async):
    route = mock_json(TASK_WEB_SEARCH)
    result = call_task(is_async, "web_search", "latest AI agent news")
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>web_search</task>" in system_msg["content"]
    assert user_msg["content"] == "latest AI agent news"
    assert result == {"results": [{"title": "AI agents", "url": "https://example.com"}]}


@IS_ASYNC
@respx.mock
def test_scrape(is_async):
    route = mock_json(TASK_SCRAPE)
    result = call_task(is_async, "scrape", ASSETS["scrape"])
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>scraper</task>" in system_msg["content"]
    assert user_msg["content"] == f"Scrape this page: {ASSETS['scrape']}"
    assert result == {"text": "Hacker News"}


@IS_ASYNC
@respx.mock
def test_translate(is_async):
    route = mock_json(TASK_TRANSLATE)
    result = call_task(is_async, "translate", "Hello there", to="French")
    body = last_body(route)
    system_msg, user_msg = body["messages"]
    assert "<task>translate</task>" in system_msg["content"]
    assert user_msg["content"] == "Translate the following into French:\n\nHello there"
    assert result == "Bonjour"


@IS_ASYNC
@respx.mock
def test_forecast_precontext_path(is_async):
    """When the forecast tool actually ran, its result surfaces via `precontext`, not
    `<task>` — forecast is model-triggered, never tagged."""
    route = mock_json(FORECAST_PRECONTEXT)
    result = call_task(is_async, "forecast", ASSETS["csv"], periods=5, unit="days")
    body = last_body(route)
    assert len(body["messages"]) == 1
    (user_msg,) = body["messages"]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == f"Forecast the next 5 days of this: {ASSETS['csv']}"
    assert result == {"forecast": [1, 2, 3]}


@IS_ASYNC
@respx.mock
def test_forecast_precontext_path_skips_unrelated_entries(is_async):
    """Other internal tools (e.g. a CSV OCR pass) may run and land in `precontext` before
    forecast does — the helper must scan past them, not just check the first entry."""
    mixed = completion(
        "Here is the forecast.",
        precontext=[
            {"name": "ocr", "result": {"extracted_text": "date,value"}},
            {"name": "forecast", "result": {"forecast": [4, 5, 6]}},
        ],
    )
    mock_json(mixed)
    result = call_task(is_async, "forecast", ASSETS["csv"])
    assert result == {"forecast": [4, 5, 6]}


@IS_ASYNC
@respx.mock
def test_forecast_fallback_path(is_async):
    """When no `forecast` precontext comes back, the helper falls back to the raw message
    content instead of raising or returning None."""
    mock_json(FORECAST_FALLBACK)
    result = call_task(is_async, "forecast", ASSETS["csv"])
    assert result == "I couldn't run the forecast tool; here's a manual estimate."
