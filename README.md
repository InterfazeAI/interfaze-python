# interfaze

The official [Interfaze](https://interfaze.ai) SDK for Python.

- **Familiar chat surface** - `chat.completions`, streaming, tools, and structured output.
- **Typed Interfaze extras** - `precontext` (internal tool output), `reasoning`, and `vcache` (semantic-cache hit) on every response.
- **One-line task helpers** - OCR, web search, scraping, speech-to-text, translation, object/GUI detection, forecasting.
- **Multimodal inputs** - images, PDFs, audio, video, and CSV, by URL or base64.
- **Sync and async**, fully typed.

## Learn more

- [interfaze.ai](https://interfaze.ai) - dashboard and API keys.
- [TypeScript / JavaScript SDK](https://github.com/InterfazeAI/interfaze-js).

## Capabilities

| Category         | Capabilities                                                |
| ---------------- | ----------------------------------------------------------- |
| **Chat & text**  | Chat completions, structured output, tools, reasoning       |
| **Vision & OCR** | `tasks.ocr` - text and structured data from images and PDFs |
| **Web**          | `tasks.web_search`, `tasks.scrape`                          |
| **Audio**        | `tasks.transcribe` - speech-to-text                         |
| **Detection**    | `tasks.object_detection`, `tasks.gui_detection`              |
| **Translation**  | `tasks.translate`                                           |
| **Forecasting**  | `tasks.forecast` - time-series prediction                   |

## Install

```bash
pip install interfaze
```

## Setup

Get an API key from the [Interfaze dashboard](https://interfaze.ai), then:

```python
from interfaze import Interfaze

interfaze = Interfaze(api_key="sk_...")  # or set INTERFAZE_API_KEY and call Interfaze()
```

Async is identical via `AsyncInterfaze` (every call becomes `await`-able).

## Usage

Chat completion:

```python
res = interfaze.chat.completions.create(
    messages=[{"role": "user", "content": "Write a haiku about deterministic AI."}],
)
print(res.choices[0].message.content)
print("cache hit:", res.vcache)          # typed Interfaze extra
```

Task helpers - each returns the extracted result directly (a `dict`/`list`/`str`, not a completion):

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

Structured output:

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

Streaming - `.stream()` yields typed events, with the inline `<think>`/`<precontext>` side-channels
stripped from the content events:

```python
stream = interfaze.chat.completions.stream(
    messages=[{"role": "user", "content": "Tell me a story."}],
)
for event in stream:
    if event.type == "content.delta":
        print(event.delta, end="")
final = stream.get_final_completion()
```

> `stream.text_deltas()` yields clean visible text only; `create(stream=True)` returns the raw
> chunk iterator (side-channel tags not stripped).

## Inputs

```python
from interfaze import inputs

inputs.image("https://…/a.png")             # image_url part
inputs.file("https://…/doc.pdf")            # file part (pdf/csv/xml/json/text/video…)
inputs.audio("https://…/a.wav")             # input_audio part
inputs.data_url(raw_bytes, "image/png")     # base64 data URI
inputs.from_path("./doc.pdf")               # read a local file
```

URLs and base64 both work; `image/gif` and `image/avif` are rejected client-side.

## Interfaze extras

- `res.precontext` - raw outputs of any internal tools that ran (OCR/web/scrape/STT/forecast/…).
- `res.reasoning` - reasoning text (with `reasoning_effort="high"` and no schema).
- `res.vcache` - whether the semantic cache was hit.
- `reasoning_effort` also accepts `"on"`/`"off"`/`"auto"`.
- Guardrails: `create(guard=["S1", "S12_IMAGE"], …)`.
- Control options: `Interfaze(show_additional_info=..., bypass_moe=..., bypass_cache=..., admin_key=...)`.

## LangChain

`pip install interfaze[langchain]` adds a chat model pointed at Interfaze that keeps the extras a stock `ChatOpenAI` drops:

```python
from interfaze.langchain import ChatInterfaze

llm = ChatInterfaze()  # reads INTERFAZE_API_KEY
res = llm.invoke("Summarize the latest AI news")
print(res.response_metadata.get("precontext"), res.response_metadata.get("vcache"))
```

`precontext`/`reasoning`/`vcache` land on `response_metadata`, `{"type": "video", ...}` content blocks are accepted, and inline side-channel tags are stripped.

## License

MIT
