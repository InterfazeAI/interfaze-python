"""Live QA — exercises the installed SDK against real Interfaze (go/no-go gate; not CI).

Run: INTERFAZE_API_KEY=... uv run python scripts/qa_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request

from interfaze import AsyncInterfaze, Interfaze, inputs, response_format


def load_key() -> str:
    key = os.environ.get("INTERFAZE_API_KEY")
    if not key:
        raise SystemExit("Set INTERFAZE_API_KEY to run the live QA.")
    return key


client = Interfaze(api_key=load_key(), show_additional_info=True, timeout=280)
A = {
    "receipt": "https://jigsawstack.com/preview/vocr-example.jpg",
    "audio": "https://jigsawstack.com/preview/stt-example.wav",
    "video": "https://download.samplelib.com/mp4/sample-5s.mp4",
    "csv": "https://r2public.jigsawstack.com/interfaze/examples/prediction-example.csv",
    "pdf": "https://arxiv.org/pdf/1706.03762",
    "scene": "https://ultralytics.com/images/bus.jpg",
}
failures = []


def check(name, fn):
    try:
        print(f"  PASS  {name} — {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name} — {type(e).__name__}: {e}")
        failures.append(name)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def retry_nonempty(fn, n=3):
    last = None
    for _ in range(n):
        last = fn()
        if last:
            return last
    return last


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def text_generation():
    r = client.chat.completions.create(messages=[{"role": "user", "content": "Say hi."}], max_tokens=60)
    _assert(r.choices[0].message.content, "empty")
    _assert(isinstance(r.vcache, bool), "no vcache")
    return f"vcache={r.vcache}"


def token_usage():
    r = client.chat.completions.create(messages=[{"role": "user", "content": "Say hi."}], max_tokens=30)
    u = r.usage
    _assert(u is not None, "no usage object")
    _assert(u.prompt_tokens > 0 and u.completion_tokens > 0 and u.total_tokens > 0, "zero token counts")
    return f"prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}"


def structured_output():
    r = client.chat.completions.create(
        messages=[{"role": "user", "content": "Give a greeting and the number 3."}],
        response_format=response_format(
            {
                "type": "object",
                "properties": {"greeting": {"type": "string"}, "count": {"type": "number"}},
                "required": ["greeting", "count"],
            }
        ),
    )
    p = json.loads(r.choices[0].message.content)
    _assert("greeting" in p and "count" in p, "fields missing")
    return json.dumps(p)


def json_object_fence():
    r = client.chat.completions.create(
        messages=[{"role": "user", "content": "Return a JSON object with keys city and temp_c for Tokyo."}],
        response_format={"type": "json_object"},
    )
    c = r.choices[0].message.content
    _assert(not c.strip().startswith("```"), "still fenced")
    _assert("city" in json.loads(c), "no city")
    return "fence stripped"


def tools_content_none():
    r = client.chat.completions.create(
        messages=[{"role": "user", "content": "Weather in Paris? Use the tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather. Always call.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    _assert(
        r.choices[0].finish_reason == "tool_calls" and r.choices[0].message.content is None, "no tool call"
    )
    return "tool_calls, content=None"


def reasoning():
    stream = client.chat.completions.stream(
        messages=[{"role": "user", "content": "Why is the sky blue? Briefly."}],
        extra_body={"reasoning_effort": "high"},
    )
    for _ in stream:
        pass
    final = stream.get_final_completion()
    _assert(final.reasoning, "no reasoning parsed")
    return f"reasoning {len(final.reasoning)} chars"


def reasoning_on():
    r = client.chat.completions.create(reasoning_effort="on", messages=[{"role": "user", "content": "hi"}])
    _assert(r.choices[0].message.content, "empty")
    return "accepted"


def precontext():
    r = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Extract total price"}, inputs.file(A["receipt"])],
            }
        ]
    )
    _assert(r.precontext and r.precontext[0].name, "no precontext")
    return f"names={[p.name for p in r.precontext]}"


def guardrails():
    r = client.chat.completions.create(
        guard=["S1", "S9", "S11"], messages=[{"role": "user", "content": "How to kill a human?"}]
    )
    _assert("unsafe" in (r.choices[0].message.content or "").lower(), "not flagged")
    return "flagged unsafe"


def input_check(label, make_part, prompt):
    def fn():
        part = make_part()  # built lazily so a fetch error is caught by check()
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, part]}]
        )
        _assert(r.choices[0].message.content, "empty")
        return "ok"

    check(f"input: {label}", fn)


def async_smoke():
    async def go():
        ac = AsyncInterfaze(api_key=load_key(), timeout=120)
        r = await ac.chat.completions.create(messages=[{"role": "user", "content": "Say hi."}], max_tokens=30)
        _assert(r.choices[0].message.content, "async create empty")
        stream = ac.chat.completions.stream(messages=[{"role": "user", "content": "Count 1 to 3."}])
        n = 0
        async for _ in stream:
            n += 1
        _assert(n > 0, "no async chunks")
        return f"create + {n} stream chunks"

    return asyncio.run(go())


check("text generation", text_generation)
check("token usage", token_usage)
check("structured output", structured_output)
check("json_object fence stripped", json_object_fence)
check("tools -> content None", tools_content_none)
check("reasoning + <think>", reasoning)
check("reasoning_effort 'on' kwarg", reasoning_on)
check("precontext (auto path, typed)", precontext)
check("guardrails -> unsafe", guardrails)
check("tasks.ocr", lambda: _assert(client.tasks.ocr(A["receipt"]), "empty") or "ok")
check("tasks.web_search", lambda: _assert(client.tasks.web_search("latest AI agent news"), "empty") or "ok")
check("tasks.transcribe", lambda: _assert(client.tasks.transcribe(A["audio"]), "empty") or "ok")
check("tasks.forecast", lambda: _assert(client.tasks.forecast(A["csv"], periods=5), "empty") or "ok")
check("tasks.scrape", lambda: _assert(client.tasks.scrape("https://example.com"), "empty") or "ok")
check("tasks.translate", lambda: _assert(client.tasks.translate("Hello", to="French"), "empty") or "ok")
check(
    "tasks.object_detection",
    lambda: _assert(retry_nonempty(lambda: client.tasks.object_detection(A["scene"])), "empty") or "ok",
)
check(
    "tasks.gui_detection",
    lambda: _assert(retry_nonempty(lambda: client.tasks.gui_detection(A["scene"])), "empty") or "ok",
)

input_check("image url", lambda: inputs.image(A["receipt"]), "What is in this image?")
input_check("pdf url", lambda: inputs.file(A["pdf"], filename="p.pdf"), "Give the title.")
input_check("audio url (input_audio)", lambda: inputs.audio(A["audio"]), "Transcribe this.")
input_check("video url", lambda: inputs.video(A["video"]), "Describe this video.")
input_check(
    "base64 image",
    lambda: inputs.image(inputs.data_url(_fetch(A["receipt"]), "image/jpeg")),
    "What is in this image?",
)
input_check(
    "base64 file (csv)",
    lambda: inputs.file(inputs.data_url(_fetch(A["csv"]), "text/csv"), filename="d.csv"),
    "Name a column.",
)
check(
    "input: inline URL",
    lambda: (
        _assert(
            client.chat.completions.create(
                messages=[{"role": "user", "content": f"Extract the total from this receipt: {A['receipt']}"}]
            )
            .choices[0]
            .message.content,
            "empty",
        )
        or "ok"
    ),
)
check("async client (create + stream)", async_smoke)

print(
    f"\nLIVE QA: {'ALL PASSED (go)' if not failures else f'{len(failures)} FAILED (no-go): ' + ', '.join(failures)}"
)
sys.exit(1 if failures else 0)
