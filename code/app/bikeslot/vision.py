"""Vision source: a rack whose camera is kept by the recognition service on the vision laptop (code/bikeslot).

That service fetches the pictures from the camera itself, decides free or occupied per hook (slot classifier or
detector, smoothed over several frames) and serves the result as JSON. The app only reads that result, so no
picture ever travels this path: it polls GET {url}/cameras/{camera}/status every few seconds (pull), or the
service posts the same JSON to POST /api/vision/{rack}/status (push: for an app on a server that the laptop can
reach but that cannot reach the laptop). Both end in ingest_vision().
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from fastapi import FastAPI

from .ingest import node_report
from .poller import describe
from .state import FREE, TAKEN, Rack, VisionSource

UNSURE = "unknown"  # the service has no decision for the slot: counted as occupied

log = logging.getLogger(__name__)
STATUS_TIMEOUT_S = 5.0
STATES = {"free": FREE, "occupied": TAKEN, "unknown": UNSURE}
SUMMARY_KEYS = ("online", "frame_age_s", "updated_at", "layout_status", "slot_count", "free", "occupied", "unknown", "processing_ms", "message", "source")


def all_taken(capacity: int, decided: dict[int, str] | None = None) -> dict:
    """A result for every space 1..capacity: the decided state where there is one, occupied everywhere else."""
    decided = decided or {}
    return {"slots": [{"id": i, "state": decided.get(i, TAKEN), "confidence": 1.0} for i in range(1, capacity + 1)]}


def vision_result(status: dict, capacity: int) -> tuple[dict, str | None]:
    """The app's detector result from the service's CameraStatus, and a warning when its numbering does not fit.

    Each slot comes as `index` 1..N with `status` free, occupied or unknown. The service has already smoothed its
    states over several frames with hysteresis, so free and occupied are passed on as decided (confidence 1).
    Without data a space never shows as free: unknown (not enough frames yet, a blurry frame, a person in front of
    the hook), a slot the service does not report and a status without slots all count as occupied."""
    slots = status.get("slots")
    if not isinstance(slots, list):
        return all_taken(capacity), None
    decided: dict[int, str] = {}
    warning = None
    for item in slots:
        try:
            index, state = int(item["index"]), STATES[item.get("status") or "unknown"]
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= index <= capacity:
            warning = f"slot {index} is outside 1 to {capacity}, the rack's capacity"
        elif state != UNSURE:
            decided[index] = state
    if warning is None and slots and len(slots) != capacity:
        warning = f"the service reports {len(slots)} slots, the rack has {capacity}"
    return all_taken(capacity, decided), warning


def vision_summary(status: dict, source: VisionSource, warning: str | None) -> dict:
    """What the admin page shows about the service's own view of this rack (its counts, not the app's)."""
    return {
        "camera": source.camera,
        "url": source.url,
        "how": "pull" if source.url else "push",
        **{key: status.get(key) for key in SUMMARY_KEYS},
        "warning": warning,
        "preview": bool(source.url and status.get("preview_url")),
    }


def untracked(store, rack: Rack, now: float) -> None:
    """No camera picture: every space counts as taken and the page shows the rack as not tracked live."""
    rack.tracking = False
    store.apply_detection(rack, all_taken(rack.capacity), 0.0, now, source="vision")


def ingest_vision(app: FastAPI, rack: Rack, status, now: float) -> tuple[dict, int]:
    """One CameraStatus from the service, polled or pushed; returns the push endpoint's reply body and HTTP status.
    A new `updated_at` is a new frame: its decided slots go through the rack (window 1, the service has smoothed
    already). An offline camera, or a picture older than STALE_AFTER_S, makes every space occupied on every status,
    so the rack never shows free spaces it cannot see. The `source` block (relay / camera health) fills the camera
    card like a node's own health report."""
    store = app.state.store
    node = rack.vision.node_id
    if not isinstance(status, dict):
        store.note_camera_error(rack, "vision: the body is not a status object", now)
        return {"ok": False, "error": "expected the CameraStatus JSON of the vision service"}, 400
    source = status.get("source") if isinstance(status.get("source"), dict) else {}
    store.note_alive(rack, node_report(source, store.settings.hot_limit_c), node, now, "vision")
    stamp = status.get("updated_at")
    fresh = stamp is not None and stamp != rack.vision_seen
    result, warning = vision_result(status, rack.capacity)
    rack.vision_state = vision_summary(status, rack.vision, warning)
    if warning:
        store.note_camera_error(rack, f"vision: {warning}", now)

    def reply(applied: bool) -> tuple[dict, int]:
        return {"ok": True, "rack": rack.id, "free": rack.free, "capacity": rack.capacity, "status": rack.status(store.settings, now), "applied": applied}, 200

    age = status.get("frame_age_s")
    if status.get("online") is False or (isinstance(age, (int, float)) and age > store.settings.stale_after_s):
        untracked(store, rack, now)   # no picture, or an old one: never shown as free spaces
        return reply(True)
    if not fresh:
        return reply(False)
    rack.vision_seen = stamp
    rack.tracking = True
    store.note_picture(rack, 0, node, now, "vision")  # a frame was decided on the laptop; it never reaches the app
    try:
        store.apply_detection(rack, result, float(status.get("processing_ms") or 0.0), now, source="vision")
    except (KeyError, TypeError, ValueError) as exc:
        store.note_camera_error(rack, f"vision: bad status: {exc}", now)
        return {"ok": False, "error": "bad vision status"}, 400
    return reply(True)


class VisionPoller:
    """Every poll_s: GET the rack's camera status from the service and hand it to ingest_vision(). A failure is
    recorded on the camera, every space counts as occupied until the service answers again, and the next round
    tries again."""

    def __init__(self, app: FastAPI, rack: Rack) -> None:
        self.app = app
        self.rack = rack
        self.vision = rack.vision

    async def poll_once(self, client: httpx.AsyncClient, now: float) -> None:
        store = self.app.state.store
        try:
            response = await client.get(self.vision.status_url, timeout=STATUS_TIMEOUT_S)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = (
                f"camera {self.vision.camera!r} is not known to the service yet (it appears with its first frame)"
                if exc.response.status_code == 404 else describe(exc)
            )
            store.note_camera_error(self.rack, f"vision: {detail}", now)
            untracked(store, self.rack, now)
            return
        except (httpx.HTTPError, ValueError) as exc:
            store.note_camera_error(self.rack, f"vision: {describe(exc)}", now)
            untracked(store, self.rack, now)
            return
        ingest_vision(self.app, self.rack, payload, now)

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await self.poll_once(client, time.time())
                except Exception as exc:  # a bug must not end the polling; the admin page shows it
                    self.app.state.store.note_camera_error(self.rack, f"vision: {describe(exc)}", time.time())
                    log.exception("polling the vision service for rack %s failed", self.rack.id)
                await asyncio.sleep(self.vision.poll_s)
