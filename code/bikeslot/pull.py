"""Pull mode: one daemon thread per camera polls a snapshot URL (BIKESLOT_PULL_CAMERAS) and processes it like a POST.

The source is the camera's own ``/snapshot`` or the relay's (relay/README.md). The view key goes as ``X-View-Key``
(a ``?key=`` is masked in logs and /health), ``If-None-Match`` skips unchanged pictures, pictures older than
``PULL_MAX_AGE_S`` are skipped, and the source's ``/health`` feeds ``CameraStatus.source`` / ``online``.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from . import config

Handle = Callable[[str, bytes], object]
Note = Callable[[str, dict], object]
# fields of the camera's own health JSON worth showing next to the slots (camera/esp32cam_ai_blur, relay /health)
HEALTH_KEYS = ("temperature_c", "wifi_rssi_dbm", "wifi_connected", "uptime_s", "pushes_ok", "pushes_failed",
               "last_push_age_s", "snapshots_failed", "last_people", "last_plates")


def redact(text: str) -> str:
    """The view key must not end up in logs or /health: ``?key=abc`` -> ``?key=***`` (URLs and error messages)."""
    return re.sub(r"(?i)([?&]key=)[^&#\s'\"]*", r"\1***", text)


def health_url(url: str) -> Optional[str]:
    """``.../snapshot?key=x`` -> ``.../health?key=x`` (the relay and the ESP32 both serve it), else None."""
    parts = urlsplit(url)
    if not parts.path.endswith("/snapshot"):
        return None
    return urlunsplit(parts._replace(path=parts.path[: -len("snapshot")] + "health"))


def summarize_health(data: dict) -> dict:
    """The relay's /health (``online``, ``last_seen_s``, ``last_snapshot_age_s``, ``health``: the camera's own) or
    the camera's own /health, reduced to what the status shows."""
    if "online" in data:
        cam = data.get("health") if isinstance(data.get("health"), dict) else {}
        out = {"kind": "relay", "online": bool(data.get("online")), "camera_id": data.get("camera_id"),
               "last_seen_s": data.get("last_seen_s"), "snapshot_age_s": data.get("last_snapshot_age_s")}
    else:
        cam = data
        out = {"kind": "camera", "online": True, "camera_id": data.get("camera_id"), "last_seen_s": 0.0,
               "snapshot_age_s": data.get("last_snapshot_age_s")}
    out.update({k: cam[k] for k in HEALTH_KEYS if k in cam})
    return out


def _age(headers) -> Optional[float]:
    try:
        return float(headers.get("X-Snapshot-Age-S"))
    except (TypeError, ValueError):
        return None


@dataclass
class Puller:
    camera_id: str
    url: str
    view_key: str = field(default="", repr=False)
    frames: int = 0
    errors: int = 0
    not_modified: int = 0        # 304 answers: the source has nothing newer than our last picture
    stale: int = 0               # pictures skipped because X-Snapshot-Age-S was above PULL_MAX_AGE_S
    state: str = "starting"      # ok | not_modified | waiting (404) | stale | error
    etag: Optional[str] = None
    last_ok: Optional[float] = None
    last_error: Optional[str] = None
    last_fetch_ms: Optional[float] = None
    source: dict = field(default_factory=dict)   # the last /health answer, summarised (summarize_health)
    stop: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def health_url(self) -> Optional[str]:
        return health_url(self.url) if config.PULL_HEALTH else None

    def info(self) -> dict:
        return {"camera_id": self.camera_id, "url": redact(self.url), "state": self.state, "frames": self.frames,
                "errors": self.errors, "not_modified": self.not_modified, "stale": self.stale,
                "last_ok_age_s": None if self.last_ok is None else round(time.time() - self.last_ok, 1),
                "last_fetch_ms": self.last_fetch_ms, "last_error": self.last_error, "source": self.source}

    def source_info(self) -> dict:
        """What the camera's status reports about where its pictures come from (``CameraStatus.source``)."""
        return {**self.source, "state": self.state, "last_error": self.last_error,
                "last_frame_age_s": None if self.last_ok is None else round(time.time() - self.last_ok, 1)}


pullers: dict[str, Puller] = {}


def parse(spec: str) -> dict[str, str]:
    """'a=http://x/snapshot, b=http://y/capture' -> {'a': ..., 'b': ...}; a bare URL becomes camera 'live01'."""
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cam, sep, url = part.partition("=")
        if not sep or cam.startswith(("http://", "https://")):
            cam, url = "live01", part
        out[cam.strip()] = url.strip()
    return out


def _problem(p: Puller, state: str, msg: str, count: bool = False) -> None:
    p.state = state
    if count:
        p.errors += 1
    msg = redact(msg)
    if msg != p.last_error:  # log each distinct problem once, not every interval
        print(f"[bikeslot] pull {p.camera_id} ({redact(p.url)}): {msg}")
    p.last_error = msg


def _fine(p: Puller, state: str) -> None:
    p.state = state
    if p.last_error:
        print(f"[bikeslot] pull {p.camera_id}: recovered")
    p.last_error = None


def _read_health(p: Puller, sess, headers: dict) -> None:
    try:
        r = sess.get(p.health_url, headers=headers, timeout=config.PULL_TIMEOUT)
        if r.status_code == 404:   # the relay knows no camera yet
            p.source = {"kind": "relay", "online": False, "note": "no data from the camera yet"}
            return
        r.raise_for_status()
        data = r.json()
        p.source = summarize_health(data) if isinstance(data, dict) else {}
    except Exception as exc:       # the picture still counts; the status just lacks the camera's health
        p.source = {**p.source, "health_error": redact(f"{type(exc).__name__}: {exc}")}


def poll_once(p: Puller, sess, handle: Handle, note: Optional[Note] = None) -> None:
    """One round: the snapshot, then the source's health. Problems are recorded on the puller, never raised."""
    t0 = time.time()
    headers = {"X-View-Key": p.view_key} if p.view_key else {}
    try:
        r = sess.get(p.url, headers={**headers, **({"If-None-Match": p.etag} if p.etag else {})},
                     timeout=config.PULL_TIMEOUT)
        p.last_fetch_ms = round((time.time() - t0) * 1000)
        age = _age(r.headers)
        too_old = age is not None and age > config.PULL_MAX_AGE_S
        if r.status_code == 304:
            p.not_modified += 1
            if too_old:
                _problem(p, "stale", f"the newest picture is {age:.0f} s old (PULL_MAX_AGE_S={config.PULL_MAX_AGE_S:g}): not used")
            else:
                _fine(p, "not_modified")
        elif r.status_code == 404:
            _problem(p, "waiting", "no picture yet (404): the camera has not sent anything to this address")
        elif r.status_code == 401:
            _problem(p, "error", "view key refused (401): check BIKESLOT_PULL_VIEW_KEY (or ?key= in BIKESLOT_PULL_CAMERAS)", count=True)
        else:
            r.raise_for_status()
            p.etag = r.headers.get("ETag")
            if too_old:
                p.stale += 1
                _problem(p, "stale", f"the newest picture is {age:.0f} s old (PULL_MAX_AGE_S={config.PULL_MAX_AGE_S:g}): not used")
            else:
                handle(p.camera_id, r.content)
                p.frames += 1
                p.last_ok = time.time()
                _fine(p, "ok")
    except Exception as exc:  # camera offline, relay down, bad JPEG, ... - keep polling
        _problem(p, "error", f"{type(exc).__name__}: {exc}", count=True)
    if p.health_url:
        _read_health(p, sess, headers)
    if note:
        note(p.camera_id, p.source_info())


def _run(p: Puller, handle: Handle, note: Optional[Note]) -> None:
    sess = requests.Session()
    while not p.stop.is_set():
        t0 = time.time()
        poll_once(p, sess, handle, note)
        p.stop.wait(max(0.0, config.PULL_INTERVAL - (time.time() - t0)))


def start(handle: Handle, spec: str = None, note: Optional[Note] = None) -> None:
    for cam, url in parse(config.PULL_CAMERAS if spec is None else spec).items():
        if cam in pullers:
            continue
        p = pullers[cam] = Puller(cam, url, view_key=config.PULL_VIEW_KEY)
        threading.Thread(target=_run, args=(p, handle, note), name=f"pull-{cam}", daemon=True).start()
        extra = " + /health" if p.health_url else ""
        print(f"[bikeslot] pulling {cam} from {redact(url)}{extra} every {config.PULL_INTERVAL:g} s"
              + (" (X-View-Key set)" if p.view_key else ""))


def stop_all() -> None:
    for p in pullers.values():
        p.stop.set()
    pullers.clear()


def info() -> list[dict]:
    return [p.info() for p in pullers.values()]
