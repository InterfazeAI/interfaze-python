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

`.stream()` yields OpenAI-style events (`content.delta`, `content.done`, tool-call events, …) with
Interfaze's inline `<think>`/`<precontext>` side-channels stripped from the content events:

```python
stream = interfaze.chat.completions.stream(
    messages=[{"role": "user", "content": "Tell me a story."}],
)
for event in stream:
    if event.type == "content.delta":
        print(event.delta, end="")
final = stream.get_final_completion()
print(final.reasoning, final.precontext)
```

> Just want clean tokens? `stream.text_deltas()` yields visible text only. Plain
> `create(stream=True)` returns the raw chunk iterator (side-channel tags **not** stripped).

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

## LangChain

`pip install interfaze[langchain]` adds a chat model that points at Interfaze and surfaces the
extras the stock `ChatOpenAI` drops:

```python
from interfaze.langchain import ChatInterfaze

llm = ChatInterfaze()  # reads INTERFAZE_API_KEY
res = llm.invoke("Summarize the latest AI news")
print(res.content)
print(res.response_metadata.get("precontext"), res.response_metadata.get("vcache"))
```

`precontext`/`reasoning`/`vcache` land on `response_metadata`; `{"type": "video", ...}` content
blocks are accepted; inline `<think>`/`<precontext>` tags are stripped. Send request-side
precontext with `ChatInterfaze(precontext=[...])`.

## Good to know

- Interfaze implements `chat.completions` and `models`; other OpenAI endpoints are not exposed.
- `temperature` ≤ 1, `max_tokens` ≤ 32000, `top_p` ≤ 1 (above → 400). Both `max_tokens` and
  `max_completion_tokens` bound output (`max_tokens` wins if both are set).
- `n`, `seed`, `stop`, penalties, `logprobs`, `tool_choice`, `top_k` are ignored by Interfaze.
- Requests default to a 900s timeout (large OCR/document/vision jobs are slow); override with
  `Interfaze(timeout=...)`.
- For very large/long documents, **stream** (`.stream()` / `create(stream=True)`): streamed
  connections are kept alive server-side, whereas a long buffered request can be dropped by an
  intermediary mid-job.
- The underlying OpenAI client is available at `interfaze.openai`.

## Compatibility notes

Drop-in for the OpenAI **chat completions** flow (`create`, response types, errors,
`create(stream=True)`, `models`), with a few behaviors worth knowing when migrating:

- **`.stream()` yields OpenAI-style events**, drop-in with OpenAI's streaming helper — iterate
  `event.type` (`"content.delta"` with `.delta`, `"content.done"`,
  `"tool_calls.function.arguments.delta"`/`.done`, …), same as `client.chat.completions.stream()`
  on the OpenAI SDK. `create(stream=True)` still gives the raw `ChatCompletionChunk` iterator, and
  `stream.text_deltas()` gives just the clean visible text if you don't need events.
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
