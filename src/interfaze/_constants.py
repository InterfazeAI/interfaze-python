from __future__ import annotations

INTERFAZE_BASE_URL = "https://api.interfaze.ai/v1"
INTERFAZE_MODEL = "interfaze-beta"
DEFAULT_TIMEOUT = 900.0

# The API error code of the keyless daily limit, and the longest Retry-After wait openai honors.
KEYLESS_DAILY_LIMIT_CODE = "keyless_daily_limit_exceeded"
MAX_HONORED_RETRY_AFTER_SECONDS = 60
NO_RETRY_AFTER_SECONDS = 0.0
SINGLE_HEADER_VALUE = 1

# Task names accepted in a <task>...</task> tag.
TASK_NAMES = (
    "ocr",
    "object_detection",
    "gui_detection",
    "web_search",
    "scraper",
    "translate",
    "speech_to_text",
    "forecast",
)

# Guardrail categories (ALL enables everything).
GUARD_CODES = (
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S1_IMAGE",
    "S12_IMAGE",
    "S15_IMAGE",
    "ALL",
)

# Formats Interfaze rejects.
BLACKLISTED_FORMATS = ("image/gif", "image/avif")

# Interfaze control-plane headers.
HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info"
HEADER_BYPASS_MOA = "x-interfaze-bypass-moa"
HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache"
HEADER_ADMIN_KEY = "x-admin-key"
