"""Real Interfaze/public asset URLs used across tests instead of fake placeholders.

None of these are fetched in tests — respx intercepts every request — but using real
URLs keeps request bodies representative of actual SDK usage.
"""

from __future__ import annotations

ASSETS = {
    "image": "https://jigsawstack.com/preview/vocr-example.jpg",
    "audio": "https://jigsawstack.com/preview/stt-example.wav",
    "csv": "https://r2public.jigsawstack.com/interfaze/examples/prediction-example.csv",
    "scene": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "gui": "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=1024",
    "pdf": "https://arxiv.org/pdf/1706.03762",
    "scrape": "https://news.ycombinator.com",
    "video": "https://download.samplelib.com/mp4/sample-5s.mp4",
}
