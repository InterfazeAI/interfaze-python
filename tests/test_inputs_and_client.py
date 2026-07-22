from __future__ import annotations

import pytest

from interfaze import AsyncInterfaze, Interfaze, InterfazeError, inputs


# ---- inputs ----
def test_image_part():
    assert inputs.image("https://x.com/a.png") == {
        "type": "image_url",
        "image_url": {"url": "https://x.com/a.png"},
    }


def test_file_part():
    part = inputs.file("https://x.com/d.pdf", filename="d.pdf")
    assert (
        part["type"] == "file"
        and part["file"]["file_data"] == "https://x.com/d.pdf"
        and part["file"]["filename"] == "d.pdf"
    )


def test_file_forwards_computed_mime():
    assert inputs.file("https://x.com/d.pdf")["file"]["format"] == "application/pdf"


def test_video_forwards_mp4_mime():
    part = inputs.video("https://x.com/clip.mp4")
    assert part["type"] == "file" and part["file"]["format"] == "video/mp4"


def test_file_unknown_ext_omits_format():
    assert "format" not in inputs.file("https://x.com/page")["file"]


def test_audio_uses_input_audio():
    part = inputs.audio("https://x.com/a.wav")
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"


def test_audio_data_uri_uses_mime_subtype():
    assert inputs.audio("data:audio/mpeg;base64,AAAA")["input_audio"]["format"] == "mpeg"
    assert inputs.audio("data:audio/wav;base64,AAAA")["input_audio"]["format"] == "wav"


def test_audio_rejects_blacklisted_data_uri():
    with pytest.raises(InterfazeError):
        inputs.audio("data:image/gif;base64,AAAA")


def test_video_uses_file_part():
    part = inputs.video("https://x.com/clip.mp4")
    assert part["type"] == "file" and part["file"]["file_data"] == "https://x.com/clip.mp4"


def test_base64_image_part():
    url = inputs.data_url(b"\x89PNG\r\n", "image/png")
    part = inputs.image(url)
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")


def test_gif_rejected():
    with pytest.raises(InterfazeError):
        inputs.image("https://x.com/a.gif")


def test_avif_rejected_via_format():
    with pytest.raises(InterfazeError):
        inputs.file("https://x.com/a", format="image/avif")


def test_data_url_base64():
    assert inputs.data_url(b"hi", "text/plain") == "data:text/plain;base64,aGk="


def test_data_url_gif_rejected():
    with pytest.raises(InterfazeError):
        inputs.data_url(b"x", "image/gif")


def test_auto_part_routing():
    assert inputs.auto_part("https://x.com/a.png")["type"] == "image_url"
    assert inputs.auto_part("https://x.com/a.wav")["type"] == "input_audio"
    assert inputs.auto_part("https://x.com/a.pdf")["type"] == "file"
    assert inputs.auto_part("https://x.com/a.mp4")["type"] == "file"
    assert inputs.auto_part("https://x.com/a.mp4")["file"]["format"] == "video/mp4"


def test_auto_part_forwards_audio_data_uri_format():
    part = inputs.auto_part("data:audio/mpeg;base64,AAAA")
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mpeg"


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
