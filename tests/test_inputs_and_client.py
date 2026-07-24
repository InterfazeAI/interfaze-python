from __future__ import annotations

import base64

import pytest
from assets import ASSETS

from interfaze import AsyncInterfaze, Interfaze, InterfazeError, inputs


# ---- inputs ----
def test_image_part():
    assert inputs.image(ASSETS["image"]) == {
        "type": "image_url",
        "image_url": {"url": ASSETS["image"]},
    }


def test_file_part():
    part = inputs.file(ASSETS["pdf"], filename="attention.pdf")
    assert (
        part["type"] == "file"
        and part["file"]["file_data"] == ASSETS["pdf"]
        and part["file"]["filename"] == "attention.pdf"
    )


def test_file_forwards_computed_mime():
    assert inputs.file(ASSETS["pdf"], filename="attention.pdf")["file"]["format"] == "application/pdf"


def test_video_forwards_mp4_mime():
    part = inputs.video(ASSETS["video"])
    assert part["type"] == "file" and part["file"]["format"] == "video/mp4"


def test_file_extensionless_omits_format():
    assert "format" not in inputs.file(ASSETS["pdf"])["file"]


def test_audio_uses_input_audio():
    part = inputs.audio(ASSETS["audio"])
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"


def test_audio_data_uri_uses_mime_subtype():
    assert inputs.audio("data:audio/mpeg;base64,AAAA")["input_audio"]["format"] == "mpeg"
    assert inputs.audio("data:audio/wav;base64,AAAA")["input_audio"]["format"] == "wav"


def test_audio_rejects_blacklisted_data_uri():
    with pytest.raises(InterfazeError):
        inputs.audio("data:image/gif;base64,AAAA")


def test_video_uses_file_part():
    part = inputs.video(ASSETS["video"])
    assert part["type"] == "file" and part["file"]["file_data"] == ASSETS["video"]


def test_base64_image_part():
    url = inputs.data_url(b"\x89PNG\r\n", "image/png")
    part = inputs.image(url)
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")


def test_file_part_with_explicit_format_included():
    part = inputs.file(ASSETS["pdf"], format="application/pdf")
    assert part["file"]["format"] == "application/pdf"


def test_video_part_rides_on_file():
    part = inputs.video(ASSETS["video"], filename="clip.mp4")
    assert part["type"] == "file"
    assert part["file"]["file_data"] == ASSETS["video"]
    assert part["file"]["filename"] == "clip.mp4"


def test_gif_rejected():
    with pytest.raises(InterfazeError):
        inputs.image(ASSETS["gif"])


def test_avif_rejected_via_format():
    with pytest.raises(InterfazeError):
        inputs.file(ASSETS["image"], format="image/avif")


def test_data_url_base64():
    assert inputs.data_url(b"hi", "text/plain") == "data:text/plain;base64,aGk="


def test_data_url_gif_rejected():
    with pytest.raises(InterfazeError):
        inputs.data_url(b"x", "image/gif")


def test_auto_part_routing():
    assert inputs.auto_part(ASSETS["image"])["type"] == "image_url"
    assert inputs.auto_part(ASSETS["audio"])["type"] == "input_audio"
    assert inputs.auto_part(ASSETS["csv"])["type"] == "file"
    assert inputs.auto_part(ASSETS["video"])["type"] == "file"
    assert inputs.auto_part(ASSETS["video"])["file"]["format"] == "video/mp4"
    assert inputs.auto_part(ASSETS["pdf"], filename="paper.pdf")["type"] == "file"


def test_auto_part_forwards_audio_data_uri_format():
    part = inputs.auto_part("data:audio/mpeg;base64,AAAA")
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mpeg"


def test_auto_part_extensionless_url_falls_through_to_file():
    """ASSETS["pdf"] (bare arxiv URL) and ASSETS["gui"] (query-string-only unsplash URL) have
    no recognizable file extension, so auto_part can't sniff a mime type and falls through to
    a generic `file` part — even for `gui`, which is actually an image."""
    assert inputs.auto_part(ASSETS["pdf"])["type"] == "file"
    assert inputs.auto_part(ASSETS["gui"])["type"] == "file"


def test_unknown_extension_file_part_has_no_format_key():
    part = inputs.file(ASSETS["svg"])
    assert "format" not in part["file"]
    assert "filename" not in part["file"]


def test_malformed_data_url_mime_none_does_not_raise():
    part = inputs.image("data:;base64,YWJj")
    assert part == {"type": "image_url", "image_url": {"url": "data:;base64,YWJj"}}


def test_from_path_reads_file_as_data_url(tmp_path):
    p = tmp_path / "note.txt"
    p.write_bytes(b"hello world")
    result = inputs.from_path(p)
    assert result.startswith("data:text/plain;base64,")
    encoded = result.split(",", 1)[1]
    assert base64.b64decode(encoded) == b"hello world"


def test_from_path_mime_by_extension(tmp_path):
    p = tmp_path / "photo.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert inputs.from_path(p).startswith("data:image/png;base64,")


def test_from_path_unknown_extension_defaults_octet_stream(tmp_path):
    p = tmp_path / "file.someunknownext"
    p.write_bytes(b"data")
    assert inputs.from_path(p).startswith("data:application/octet-stream;base64,")


def test_from_path_blacklisted_ext_raises(tmp_path):
    p = tmp_path / "anim.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(InterfazeError):
        inputs.from_path(p)


def test_from_path_accepts_str_path(tmp_path):
    p = tmp_path / "note.md"
    p.write_bytes(b"# hi")
    assert inputs.from_path(str(p)).startswith("data:text/markdown;base64,")


# ---- client surface ----
def test_curated_surface():
    c = Interfaze(api_key="t")
    assert hasattr(c, "chat") and hasattr(c, "models") and hasattr(c, "tasks")
    assert not hasattr(c, "embeddings") and not hasattr(c, "responses")
    assert hasattr(c, "openai")  # escape hatch


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    with pytest.raises(InterfazeError, match="INTERFAZE_API_KEY"):
        Interfaze()


def test_default_timeout_covers_server_cap():
    assert Interfaze(api_key="t").openai.timeout == 900.0
    assert AsyncInterfaze(api_key="t").openai.timeout == 900.0


def test_timeout_override_respected():
    assert Interfaze(api_key="t", timeout=30.0).openai.timeout == 30.0
