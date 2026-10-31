"""One path for what a camera node sends, whether it posts to the app (push) or the app polls it (pull)."""

from __future__ import annotations

from fastapi import FastAPI

from .detector_client import DetectorError, fake_result
from .history import parse_clock
from .state import Rack

NODE_INTS = (
    "wifi_rssi_dbm", "wifi_channel", "heap_free", "heap_min_free", "heap_largest_block", "psram_free", "psram_total",
    "stream_frames", "stream_dropped",
)


def node_report(payload: dict, hot_limit_c: float) -> dict:
    """Normalise what a node says about itself into Camera field names. Understands the node's status JSON
    (temperature_c, too_hot, uptime_s, WiFi and memory figures) and the push contract (tempC, tooHot, uptimeS).
    Without a too_hot flag the app compares the temperature with the hot limit and marks the result as derived."""
    report: dict = {}
    temp = payload.get("temperature_c", payload.get("tempC"))
    if temp is not None:
        temp = report["temp_c"] = float(temp)
    hot = payload.get("too_hot", payload.get("tooHot"))
    if hot is not None:
        report["too_hot"], report["too_hot_derived"] = bool(hot), False
    else:
        report["too_hot"] = temp is not None and temp > hot_limit_c
        report["too_hot_derived"] = temp is not None
    uptime = payload.get("uptime_s", payload.get("uptimeS"))
    if uptime is not None:
        report["uptime_s"] = int(uptime)
    for key in NODE_INTS:
        if payload.get(key) is not None:
            report[key] = int(payload[key])
    if payload.get("wifi_connected") is not None:
        report["wifi_connected"] = bool(payload["wifi_connected"])
    if payload.get("ip") is not None:
        report["ip"] = str(payload["ip"])
    return report


def ingest_alive(app: FastAPI, rack: Rack, payload: dict, node_id: str | None, now: float, source: str) -> None:
    app.state.store.note_alive(rack, node_report(payload, app.state.settings.hot_limit_c), node_id, now, source)


async def ingest_picture(
    app: FastAPI, rack: Rack, image: bytes, node_id: str | None, now: float, source: str, demo_clock: str | None = None
) -> tuple[dict, int]:
    """Run one picture through the detector (the stand-in for demo racks) and then the rules. Returns the reply
    body and HTTP status the push endpoint answers with; the poller has nobody to answer and drops them.
    `demo_clock` is the demo driver's simulated time (X-Demo-Clock); only a demo rack uses it."""
    store = app.state.store
    store.note_picture(rack, len(image), node_id, now, source)
    try:
        if rack.demo:  # made-up pictures are decided by the stand-in, the real detector never sees them
            result, latency_ms = fake_result(rack.id, rack.capacity, image), 0.0
        else:
            result, latency_ms = await app.state.detector.analyze(rack.id, rack.capacity, image)
    except DetectorError as exc:
        store.note_detector_error(str(exc), now)
        return {"ok": False, "error": "detector unavailable"}, 503
    finally:
        del image  # not kept beyond this request
    try:
        store.apply_detection(rack, result, latency_ms, now, parse_clock(demo_clock) if rack.demo else None)
    except (KeyError, TypeError, ValueError) as exc:
        store.note_detector_error(f"bad detector result: {exc}", now)
        return {"ok": False, "error": "bad detector result"}, 502
    # not rack_view: its forecast is not needed here and would be worked out again for every picture
    return {"ok": True, "rack": rack.id, "free": rack.free, "capacity": rack.capacity, "status": rack.status(store.settings, now)}, 200
