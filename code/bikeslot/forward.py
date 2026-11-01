"""Push mode towards the app (code/app): every new camera status is posted to the app's vision endpoint.

Only for an app that cannot reach this service (otherwise the app polls). Off unless BIKESLOT_APP_URL is set;
BIKESLOT_APP_RACKS maps camera -> rack, BIKESLOT_APP_TOKEN is sent as X-Vision-Token. One thread posts only the
newest status per camera, so a slow app never delays frame processing. State: ``GET /health`` -> "forward".
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

from . import config


def parse_racks(spec: str) -> dict[str, str]:
    """'live01=MX-CAM, cam02=B' -> {'live01': 'MX-CAM', 'cam02': 'B'}; malformed parts are ignored."""
    out: dict[str, str] = {}
    for part in spec.split(","):
        cam, sep, rack = part.strip().partition("=")
        if sep and cam.strip() and rack.strip():
            out[cam.strip()] = rack.strip()
    return out


@dataclass
class Target:
    camera_id: str
    rack_id: str
    url: str
    sent: int = 0
    errors: int = 0
    last_ok: Optional[float] = None
    last_error: Optional[str] = None
    last_ms: Optional[float] = None
    last_reply: Optional[dict] = None   # the app's answer: its free count and rack status after applying ours

    def info(self) -> dict:
        return {"camera_id": self.camera_id, "rack_id": self.rack_id, "url": self.url, "sent": self.sent, "errors": self.errors,
                "last_ok_age_s": None if self.last_ok is None else round(time.time() - self.last_ok, 1),
                "last_ms": self.last_ms, "last_error": self.last_error, "last_reply": self.last_reply}


class Forwarder:
    def __init__(self, app_url: str, racks, token: str = "", timeout: float = 5.0,
                 post: Optional[Callable[[str, dict, dict], tuple[int, object]]] = None):
        self.app_url = (app_url or "").strip().rstrip("/")
        mapping = parse_racks(racks) if isinstance(racks, str) else dict(racks)
        self.targets: dict[str, Target] = (
            {cam: Target(cam, rack, f"{self.app_url}/api/vision/{rack}/status") for cam, rack in mapping.items()} if self.app_url else {})
        self.token = token or ""
        self.timeout = timeout
        self._post = post or self._http_post
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._session: Optional[requests.Session] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return bool(self.targets)

    # ------------------------------------------------------------------ producer side (frame processing)
    def submit(self, camera_id: str, status) -> bool:
        """Queue the newest status of a camera; returns False when the camera is not forwarded."""
        if camera_id not in self.targets:
            return False
        payload = status.model_dump(mode="json") if hasattr(status, "model_dump") else dict(status)
        with self._lock:
            self._pending[camera_id] = payload
        self._wake.set()
        return True

    # ------------------------------------------------------------------ consumer side (background thread)
    def flush(self) -> int:
        """Send everything pending now (the thread calls this; tests call it directly). Returns how many were sent."""
        with self._lock:
            batch, self._pending = self._pending, {}
        for cam, payload in batch.items():
            self._send(self.targets[cam], payload)
        return len(batch)

    def _send(self, t: Target, payload: dict) -> None:
        t0 = time.time()
        try:
            code, body = self._post(t.url, payload, self._headers())
            t.last_ms = round((time.time() - t0) * 1000)
            if code >= 400:
                raise RuntimeError(f"HTTP {code}: {str(body)[:200]}")
            t.sent += 1
            t.last_ok = time.time()
            t.last_reply = body if isinstance(body, dict) else None
            if t.last_error:
                print(f"[bikeslot] forward {t.camera_id} -> {t.url}: recovered")
            t.last_error = None
        except Exception as exc:  # the app being down must never affect the frame processing
            t.errors += 1
            msg = str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}"
            if msg != t.last_error:   # log each distinct error once, not every frame
                print(f"[bikeslot] forward {t.camera_id} -> {t.url} failed: {msg}")
            t.last_error = msg

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Vision-Token"] = self.token
        return headers

    def _http_post(self, url: str, payload: dict, headers: dict) -> tuple[int, object]:
        if self._session is None:
            self._session = requests.Session()
        r = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
        try:
            body = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            self.flush()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="forward-app", daemon=True)
        self._thread.start()
        for t in self.targets.values():
            print(f"[bikeslot] forwarding {t.camera_id} -> {t.url}" + (" (with token)" if self.token else ""))

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def info(self) -> dict:
        return {"app_url": self.app_url or None, "targets": [t.info() for t in self.targets.values()]}


forwarder = Forwarder("", {})   # replaced by start() at service start


def start() -> Forwarder:
    global forwarder
    forwarder = Forwarder(config.APP_URL, config.APP_RACKS, config.APP_TOKEN, config.APP_TIMEOUT)
    forwarder.start()
    return forwarder


def submit(camera_id: str, status) -> bool:
    return forwarder.submit(camera_id, status)


def stop() -> None:
    forwarder.stop()


def info() -> dict:
    return forwarder.info()
