# interfaze

The official [Interfaze](https://interfaze.ai) SDK for Python — a thin, typed wrapper over the
OpenAI SDK. Same `chat.completions` surface, plus typed access to everything Interfaze adds
(`precontext`, `reasoning`, `vcache`, `task`/`guard` helpers). Sync and async.

## Install

```bash
pip install interfaze
export INTERFAZE_API_KEY="sk_..."
```

## Quickstart

```python
from interfaze import Interfaze

interfaze = Interfaze()  # reads INTERFAZE_API_KEY

res = interfaze.chat.completions.create(
    messages=[{"role": "user", "content": "Write a haiku about deterministic AI."}],
)
print(res.choices[0].message.content)
print("cache hit:", res.vcache)          # typed Interfaze extra
```

Async:

```python
from interfaze import AsyncInterfaze

interfaze = AsyncInterfaze()
res = await interfaze.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])
```

## Task helpers

```python
interfaze.tasks.ocr("https://example.com/receipt.jpg")
interfaze.tasks.web_search("latest AI agent news")
interfaze.tasks.transcribe("https://example.com/audio.wav")
interfaze.tasks.scrape("https://example.com/product")
interfaze.tasks.translate("Hello", to="French")
interfaze.tasks.object_detection("https://example.com/photo.jpg")
interfaze.tasks.gui_detection("https://example.com/screenshot.png")
interfaze.tasks.forecast("https://example.com/series.csv", periods=30)
```

Or force a task on a raw completion:

```python
from interfaze import inputs

res = interfaze.chat.completions.create(
    task="ocr",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Extract the total"},
        inputs.file("https://example.com/receipt.jpg"),
    ]}],
)
```

## Structured output

```python
from interfaze import response_format

res = interfaze.chat.completions.create(
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    response_format=response_format({
        "type": "object",
        "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
        "required": ["city", "temp_c"],
    }),
)
```

## Streaming

```python
stream = interfaze.chat.completions.stream(
    messages=[{"role": "user", "content": "Tell me a story."}],
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
final = stream.get_final_completion()
print(final.reasoning, final.precontext)
```

> Plain `create(stream=True)` also works and returns the raw chunk iterator; `.stream()` adds
> accumulation and surfaces `reasoning`/`precontext`.

## Inputs

```python
from interfaze import inputs

inputs.image("https://…/a.png")             # image_url part
inputs.file("https://…/doc.pdf")            # file part (pdf/csv/xml/json/text/video…)
inputs.audio("https://…/a.wav")             # input_audio part
inputs.data_url(raw_bytes, "image/png")     # base64 data URI
inputs.from_path("./doc.pdf")               # read a local file
```

URLs and base64 work; raw `bytes` do **not** (must be base64-encoded — this SDK does it for you via
`data_url`/`from_path`). `image/gif` and `image/avif` are rejected client-side.

## Interfaze extras

- `res.precontext` — raw outputs of internal tools that ran (OCR/web/scrape/STT/forecast/…).
- `res.reasoning` — reasoning text (with `reasoning_effort="high"` and no schema).
- `res.vcache` — whether the semantic cache was hit.
- `reasoning_effort` accepts `"on"`/`"off"`/`"auto"` in addition to `minimal|low|medium|high`.
- Guardrails: `create(guard=["S1", "S12_IMAGE"], …)`.
- Control options: `Interfaze(show_additional_info=..., bypass_moe=..., bypass_cache=..., admin_key=...)`.
- Custom params: pass `extra_body={...}` / `extra_headers={...}` straight through to the request.

## Good to know

- Interfaze implements `chat.completions` and `models`; other OpenAI endpoints are not exposed.
- `temperature` ≤ 1, `max_tokens` ≤ 32000, `top_p` ≤ 1 (above → 400). Use `max_tokens` (not
  `max_completion_tokens`) to bound output.
- `n`, `seed`, `stop`, penalties, `logprobs`, `tool_choice`, `top_k` are ignored by Interfaze.
- The underlying OpenAI client is available at `interfaze.openai`.

## Compatibility notes

Drop-in for the OpenAI **chat completions** flow (`create`, response types, errors,
`create(stream=True)`, `models`), with a few behaviors worth knowing when migrating:

- **`.stream()` yields chunks, not events.** OpenAI's `.stream()` helper yields event objects
  (`event.type == "content.delta"`); ours yields raw `ChatCompletionChunk`s (like
  `create(stream=True)`) plus `get_final_completion()`. Iterate `chunk.choices[0].delta`, not
  `event.type`.
- **Returned text is lightly post-processed.** `json_object` content is unwrapped from its
  ```` ```json ```` fence, and streamed `<think>`/`<precontext>` side-channels are pulled into
  `reasoning`/`precontext`, so `message.content` may not be byte-identical to the raw wire response.
- **`inputs.*` accept https URLs** (Interfaze fetches them server-side). Those parts are valid for
  Interfaze but **not** portable to OpenAI/Azure, which require base64 in `file`/`input_audio` parts.
- **Escape hatch:** anything not on the wrapper — `chat.completions.parse()`, `.with_raw_response`,
  `.with_streaming_response` — is on the underlying client at `interfaze.openai`.
- **`tasks.*` return the extracted result** (a `dict`/`list`/`str`), not a `ChatCompletion` — e.g.
  `tasks.ocr(...)` returns the OCR dict directly.

## License

MIT
