# Interfaze Python SDK

The official [Interfaze](https://interfaze.ai) SDK for Python

[Docs](https://interfaze.ai/docs) · [limits](https://interfaze.ai/docs/limits) · [pricing](https://interfaze.ai/pricing) · [dashboard](https://interfaze.ai) · [TypeScript / JavaScript SDK](https://github.com/InterfazeAI/interfaze-js)

## Install

```bash
pip install interfaze
# or: uv add interfaze · poetry add interfaze
```

## Setup

```python
from interfaze import Interfaze

interfaze = Interfaze(api_key="sk_...")  # or set INTERFAZE_API_KEY and call Interfaze()
```

Async is identical via `AsyncInterfaze` - every call becomes `await`-able.

## Your first request

This guide will get you started with your first request to Interfaze, which follows the Chat Completions API standard.

```python
import json
from interfaze import Interfaze, response_format

interfaze = Interfaze()

res = interfaze.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the details from this ID."},
                {"type": "image_url", "image_url": {"url": "https://r2public.jigsawstack.com/interfaze/examples/id.jpg"}},
            ],
        },
    ],
    response_format=response_format(
        {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "dob": {"type": "string", "description": "Date of birth on the ID"},
                "licence_number": {"type": "string"},
            },
            "required": ["first_name", "last_name", "dob", "licence_number"],
        },
        "id_card",
    ),
)

id_card = json.loads(res.choices[0].message.content or "{}")
print(id_card)
print("OCR result:", res.precontext[0].result if res.precontext else None)  # the raw OCR that produced it
```

## Precontext

Alongside the answer, a response carries `precontext` - the raw metadata Interfaze produced while answering:

```python
for p in res.precontext or []:
    print(p.name, p.result)  # e.g. "ocr" -> { text, boxes, confidence, … }
```

## Chat

```python
res = interfaze.chat.completions.create(
    messages=[{"role": "user", "content": "Which US public companies reported earnings today?"}],
)

res.choices[0].message.content
```

The result is a standard `ChatCompletion` with `precontext` and `reasoning` added (a web search backs the answer here).

### Streaming

Stream the reply as it's generated; the final completion still carries `precontext` and `reasoning`.

```python
stream = interfaze.chat.completions.stream(
    messages=[{"role": "user", "content": "Summarize this week's top AI research and cite your sources."}],
)

for text in stream.text_deltas():
    print(text, end="", flush=True)

final = stream.get_final_completion()  # .precontext (the sources), .reasoning
```

`text_deltas()` yields display-ready text - the inline `<think>`/`<precontext>` side-channels are stripped and returned structured on `get_final_completion()`. Iterate the stream directly (`for event in stream`) for typed events, or `create(stream=True)` for the raw chunk iterator.

### Structured output

`response_format()` takes a JSON Schema and normalizes it for Interfaze:

```python
import json
from interfaze import inputs, response_format

res = interfaze.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract this receipt."},
                inputs.image("https://jigsawstack.com/preview/vocr-example.jpg"),
            ],
        },
    ],
    response_format=response_format(
        {
            "type": "object",
            "properties": {
                "merchant": {"type": "string"},
                "total": {"type": "number"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
                    },
                },
            },
            "required": ["merchant", "total", "items"],
        },
        "receipt",
    ),
)

receipt = json.loads(res.choices[0].message.content or "{}")  # {"merchant": ..., "total": ..., "items": [...]}
```

`message.content` comes back as a JSON string, so parse it - and keep the root an `object`, since a non-object root is wrapped under a `result` key. Prefer Pydantic? Use `interfaze.chat.completions.parse(response_format=YourModel, …)` and read `choices[0].message.parsed`. The result is an `InterfazeParsedChatCompletion`, so `precontext` (a list of `Precontext`, e.g. `res.precontext[0].result`) and `vcache` are typed just like on `create()`.

### Tools and function calling

Interfaze supports tools - define `tools`, read `message.tool_calls`, run them, then pass the results back for the final answer.

```python
import json

def get_weather(city: str) -> str:
    return json.dumps({"city": city, "temp_c": 21, "condition": "clear"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "e.g. Tokyo"}},
                "required": ["city"],
            },
        },
    },
]

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
res = interfaze.chat.completions.create(messages=messages, tools=tools, tool_choice="auto")

message = res.choices[0].message
messages.append(message.model_dump())  # the assistant turn, carrying any tool_calls

for call in message.tool_calls or []:
    args = json.loads(call.function.arguments)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": get_weather(args["city"])})

final = interfaze.chat.completions.create(messages=messages, tools=tools, tool_choice="auto")
```

## Reasoning

Ask for reasoning with `reasoning_effort`; the text comes back on `res.reasoning`.

```python
res = interfaze.chat.completions.create(
    reasoning_effort="high",  # also accepts Interfaze's "on" / "off" / "auto"
    messages=[{"role": "user", "content": "Which region should we launch in first, and why?"}],
)

res.reasoning  # reasoning text - present with reasoning_effort and no schema
```

## Multimodal Inputs

Interfaze handles images, PDFs, audio, video, and CSV. The simplest way is to drop a public URL into the prompt - Interfaze fetches and reads it:

```python
interfaze.chat.completions.create(
    messages=[{"role": "user", "content": "Summarize this document: https://arxiv.org/pdf/1706.03762"}],
)
```

Or attach it as a content part - a `file` part for documents, audio, and video; `image_url` for images:

```python
interfaze.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this document."},
                {"type": "file", "file": {"filename": "paper.pdf", "file_data": "https://arxiv.org/pdf/1706.03762"}},
            ],
        },
    ],
)
```

`file_data` also takes a base64 data URI:

```python
{"type": "file", "file": {"filename": "report.pdf", "file_data": f"data:application/pdf;base64,{base64}"}}
```

`inputs.*` is a typed shortcut that builds these parts - and turns raw bytes or a local file into a data URI for you:

```python
from interfaze import inputs

inputs.image("https://…/photo.png")                                       # image_url part
inputs.file("https://…/report.pdf")                                       # file part
inputs.audio("https://…/call.wav")                                        # input_audio part
inputs.file(inputs.data_url(pdf_bytes, "application/pdf"), filename="report.pdf")  # from raw bytes
inputs.image(inputs.from_path("./photo.png"))                             # read a local file
```

Interfaze rejects `image/gif` and `image/avif` client-side.

## Tasks

A task ([run_task](https://interfaze.ai/docs/run-tasks)) runs one built-in tool instead of the full model - faster and cheaper, but limited to that tool and its fixed output structure. So `task` can't be combined with a custom `response_format`; reach for a full completion (like [your first request](#your-first-request)) when you need the whole model or your own schema.

The `tasks.*` helpers are the shortest way - each takes a source and returns the raw result:

```python
interfaze.tasks.ocr(url)
interfaze.tasks.object_detection(url)
interfaze.tasks.gui_detection(url)
interfaze.tasks.web_search(query)
interfaze.tasks.scrape(url)
interfaze.tasks.transcribe(url)
interfaze.tasks.translate(text, to="Spanish")
interfaze.tasks.forecast(csv_url, periods=30, unit="days")
```

For a multi-part message (text plus an input), set `task` on a normal `create` call - the result comes back on `message.content`:

```python
import json

res = interfaze.chat.completions.create(
    task="ocr",
    messages=[{"role": "user", "content": [{"type": "text", "text": "Extract the total."}, inputs.image(url)]}],
)
result = json.loads(res.choices[0].message.content or "{}").get("result")  # same output as tasks.ocr()
```

Or a `<task>…</task>` system message plus an empty response schema:

```python
import json
from interfaze import empty_task_schema

res = interfaze.chat.completions.create(
    messages=[
        {"role": "system", "content": "<task>ocr</task>"},
        {"role": "user", "content": [{"type": "text", "text": "Extract the total."}, inputs.image(url)]},
    ],
    response_format=empty_task_schema(),  # an empty JSON schema
)
result = json.loads(res.choices[0].message.content or "{}").get("result")
```

## Guardrails

Enable safety categories with `guard`; a blocked request comes back as a normal completion, not an exception.

```python
res = interfaze.chat.completions.create(
    guard=["S1", "S10", "S12_IMAGE"],
    messages=[{"role": "user", "content": "..."}],
)
```

A match returns the plain string `unsafe S1` as `message.content` - so check for it. See the exported `GUARD_CODES`.

## Client options

Set router, cache, and streaming behavior once on the client:

```python
interfaze = Interfaze(
    show_additional_info=True,  # stream <precontext> deltas as they're produced
    bypass_moa=True,            # skip the mixture-of-architecture router
    bypass_cache=True,          # skip the semantic cache
)
```

## Errors

Interfaze re-exports typed error classes to catch and narrow on:

```python
from interfaze import BadRequestError, InterfazeError, RateLimitError
```

`InterfazeError` is client-side (missing key, invalid guard code, stream misuse). Everything else is an `APIError` subclass carrying `status_code` and `code` - `BadRequestError` (400), `AuthenticationError` (401), `RateLimitError` (429), and so on.

## Capabilities

| Use case                                | Entry point                                     |
| --------------------------------------- | ----------------------------------------------- |
| [Chat](#chat)                           | `chat.completions.create`                       |
| [Streaming](#streaming)                 | `chat.completions.stream`                       |
| [Structured output](#structured-output) | `response_format()`                             |
| [Reasoning](#reasoning)                 | `reasoning_effort`                              |
| [Tools](#tools-and-function-calling)    | `tools`                                         |
| [Multimodal inputs](#multimodal-inputs) | `inputs.*`                                      |
| [OCR](#tasks)                           | `tasks.ocr`                                     |
| [Object and GUI detection](#tasks)      | `tasks.object_detection`, `tasks.gui_detection` |
| [Web search and scraping](#tasks)       | `tasks.web_search`, `tasks.scrape`              |
| [Speech to text](#tasks)                | `tasks.transcribe`                              |
| [Translation](#tasks)                   | `tasks.translate`                               |
| [Forecasting](#tasks)                   | `tasks.forecast`                                |
| [Guardrails](#guardrails)               | `guard`                                         |
| [Precontext](#precontext)               | `res.precontext`                                |

## License

MIT
