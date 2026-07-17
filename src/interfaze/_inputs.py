from __future__ import annotations

import base64 as _base64
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ._constants import BLACKLISTED_FORMATS
from ._errors import InterfazeError

BytesLike = Union[bytes, bytearray]

_EXT_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
    "gif": "image/gif", "bmp": "image/bmp", "heic": "image/heic", "heif": "image/heif",
    "pdf": "application/pdf", "csv": "text/csv", "tsv": "text/tab-separated-values",
    "xml": "application/xml", "json": "application/json", "txt": "text/plain",
    "md": "text/markdown", "markdown": "text/markdown", "yaml": "application/yaml", "yml": "application/yaml",
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac",
    "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm", "avi": "video/x-msvideo",
    "mkv": "video/x-matroska", "3gp": "video/3gpp",
}


def _mime_from_data_url(s: str) -> Optional[str]:
    if s.startswith("data:"):
        return s[5:].split(";")[0].split(",")[0] or None
    return None


def _ext_of(url_or_name: str) -> Optional[str]:
    base = url_or_name.split("?")[0].split("#")[0]
    return base.rsplit(".", 1)[-1].lower() if "." in base else None


def _assert_allowed(mime: Optional[str]) -> None:
    if mime and mime in BLACKLISTED_FORMATS:
        raise InterfazeError(f'Format "{mime}" is not supported by Interfaze.')


def data_url(data: BytesLike, mime_type: str) -> str:
    """Build a base64 ``data:`` URI from raw bytes."""
    _assert_allowed(mime_type)
    return f"data:{mime_type};base64,{_base64.b64encode(bytes(data)).decode('ascii')}"


def from_path(path: Union[str, Path]) -> str:
    """Read a local file into a ``data:`` URI (mime by extension)."""
    p = Path(path)
    mime = _EXT_MIME.get((p.suffix.lstrip(".")).lower(), "application/octet-stream")
    return data_url(p.read_bytes(), mime)


def image(src: str) -> Dict[str, Any]:
    """Image content part. ``src`` = https URL or ``data:`` URI."""
    _assert_allowed(_mime_from_data_url(src) or _EXT_MIME.get(_ext_of(src) or ""))
    return {"type": "image_url", "image_url": {"url": src}}


def file(src: str, *, filename: Optional[str] = None, format: Optional[str] = None) -> Dict[str, Any]:
    """File content part (pdf/csv/xml/json/text/audio/video/…). ``src`` = https URL or ``data:`` URI."""
    mime = format or _mime_from_data_url(src) or _EXT_MIME.get(_ext_of(filename or src) or "")
    _assert_allowed(mime)
    f: Dict[str, Any] = {"file_data": src}
    if filename:
        f["filename"] = filename
    if format:
        f["format"] = format
    return {"type": "file", "file": f}


def audio(src: str, *, format: Optional[str] = None) -> Dict[str, Any]:
    """Audio content part via ``input_audio`` (``audio_url`` is a dead field in Interfaze)."""
    fmt = format or _ext_of(src) or "wav"
    return {"type": "input_audio", "input_audio": {"data": src, "format": fmt}}


def video(src: str, *, filename: Optional[str] = None) -> Dict[str, Any]:
    """Video content part — rides on the ``file`` part (there is no video content part)."""
    return file(src, filename=filename)


def auto_part(src: str, *, filename: Optional[str] = None, format: Optional[str] = None) -> Dict[str, Any]:
    """Pick a content part by media type: image → image_url, audio → input_audio, else file."""
    mime = format or _mime_from_data_url(src) or _EXT_MIME.get(_ext_of(filename or src) or "")
    if mime and mime.startswith("image/"):
        return image(src)
    if mime and mime.startswith("audio/"):
        return audio(src, format=format) if format else audio(src)
    return file(src, filename=filename, format=format)
