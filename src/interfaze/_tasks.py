from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from ._chat import AsyncCompletions, Completions
from ._inputs import auto_part
from ._types import TaskName

Content = Union[str, List[Dict[str, Any]]]


def _text(t: str) -> Dict[str, Any]:
    return {"type": "text", "text": t}


def _extract(content: Optional[str]) -> Any:
    if not content:
        return None
    try:
        parsed = json.loads(content)
        return parsed.get("result", parsed) if isinstance(parsed, dict) else parsed
    except (ValueError, TypeError):
        return content


def _forecast_prompt(csv_source: str, periods: int, unit: str) -> str:
    return f"Forecast the next {periods} {unit} of this: {csv_source}"


class Tasks:
    """High-level task helpers (sync). Each returns the task's raw ``result``."""

    def __init__(self, completions: Completions) -> None:
        self._c = completions

    def _run(self, task: TaskName, content: Content) -> Any:
        r = self._c.create(task=task, messages=[{"role": "user", "content": content}])
        return _extract(r.choices[0].message.content)

    def ocr(self, source: str, *, prompt: str = "Extract all text and data.") -> Any:
        return self._run("ocr", [_text(prompt), auto_part(source)])

    def object_detection(self, source: str, *, prompt: str = "Detect all objects.") -> Any:
        return self._run("object_detection", [_text(prompt), auto_part(source)])

    def gui_detection(self, source: str, *, prompt: str = "Detect all GUI elements.") -> Any:
        return self._run("gui_detection", [_text(prompt), auto_part(source)])

    def transcribe(self, source: str, *, prompt: str = "Transcribe this audio.") -> Any:
        return self._run("speech_to_text", [_text(prompt), auto_part(source)])

    def web_search(self, query: str) -> Any:
        return self._run("web_search", query)

    def scrape(self, url: str, *, prompt: str = "Scrape this page") -> Any:
        return self._run("scraper", f"{prompt}: {url}")

    def translate(self, text: str, *, to: str) -> Any:
        return self._run("translate", f"Translate the following into {to}:\n\n{text}")

    def forecast(self, csv_source: str, *, periods: int = 10, unit: str = "days") -> Any:
        r = self._c.create(messages=[{"role": "user", "content": _forecast_prompt(csv_source, periods, unit)}])
        for p in r.precontext or []:
            if p.name == "forecast":
                return p.result
        return r.choices[0].message.content


class AsyncTasks:
    """High-level task helpers (async)."""

    def __init__(self, completions: AsyncCompletions) -> None:
        self._c = completions

    async def _run(self, task: TaskName, content: Content) -> Any:
        r = await self._c.create(task=task, messages=[{"role": "user", "content": content}])
        return _extract(r.choices[0].message.content)

    async def ocr(self, source: str, *, prompt: str = "Extract all text and data.") -> Any:
        return await self._run("ocr", [_text(prompt), auto_part(source)])

    async def object_detection(self, source: str, *, prompt: str = "Detect all objects.") -> Any:
        return await self._run("object_detection", [_text(prompt), auto_part(source)])

    async def gui_detection(self, source: str, *, prompt: str = "Detect all GUI elements.") -> Any:
        return await self._run("gui_detection", [_text(prompt), auto_part(source)])

    async def transcribe(self, source: str, *, prompt: str = "Transcribe this audio.") -> Any:
        return await self._run("speech_to_text", [_text(prompt), auto_part(source)])

    async def web_search(self, query: str) -> Any:
        return await self._run("web_search", query)

    async def scrape(self, url: str, *, prompt: str = "Scrape this page") -> Any:
        return await self._run("scraper", f"{prompt}: {url}")

    async def translate(self, text: str, *, to: str) -> Any:
        return await self._run("translate", f"Translate the following into {to}:\n\n{text}")

    async def forecast(self, csv_source: str, *, periods: int = 10, unit: str = "days") -> Any:
        r = await self._c.create(messages=[{"role": "user", "content": _forecast_prompt(csv_source, periods, unit)}])
        for p in r.precontext or []:
            if p.name == "forecast":
                return p.result
        return r.choices[0].message.content
