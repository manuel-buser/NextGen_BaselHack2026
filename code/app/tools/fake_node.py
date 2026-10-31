"""Stand-in for a camera node in pull mode: serves the node's status JSON and a picture for the app to fetch.

Run from code/app: uvicorn tools.fake_node:app --port 8010, then give a rack in the racks file
camera: {url: "http://localhost:8010"} and start the app. The temperature drifts a little and the uptime counts.
The picture is synthetic bytes whose length steers the app's built-in stand-in detector (same rule as
tools/fake_camera.py); the taken count random-walks between 0 and CAPACITY. No too_hot field: the app derives it.
"""

from __future__ import annotations

import random
import time

from fastapi import FastAPI, Response

CAPACITY = 15


def make_app(taken: int = 5) -> FastAPI:
    """A fresh node: the first status reply says 51.3 C, the first picture has `taken` spaces taken."""
    app = FastAPI(title="Fake camera node")
    node = {"started": time.monotonic(), "temp_c": 51.3, "taken": taken}

    @app.get("/status")
    async def status() -> dict:
        reply = {
            "uptime_s": int(time.monotonic() - node["started"]),
            "temperature_c": node["temp_c"],
            "stream_active": False,
            "stream_frames": 0,
            "stream_dropped": 0,
            "heap_free": 202840,
            "heap_min_free": 198216,
            "heap_largest_block": 159732,
            "psram_free": 7987060,
            "psram_total": 8388608,
            "wifi_connected": True,
            "wifi_rssi_dbm": -38,
            "wifi_channel": 5,
            "ip": "127.0.0.1",
        }
        node["temp_c"] = round(min(60.0, max(45.0, node["temp_c"] + random.uniform(-0.3, 0.3))), 1)  # drift after the reply
        return reply

    @app.get("/capture")
    async def capture() -> Response:
        picture = b"FAKEJPEG" + b"\0" * (100 * (CAPACITY + 1) + node["taken"] - 8)
        node["taken"] = min(CAPACITY, max(0, node["taken"] + random.choice((-1, 0, 0, 1))))
        return Response(content=picture, media_type="image/jpeg")

    return app


app = make_app()
