"""Pull mode: the service polls camera snapshot URLs itself (no separate replay process needed).

Configured with ``BIKESLOT_PULL_CAMERAS="cam_id=http://ip/snapshot,other=http://ip2/capture"``.
One daemon thread per camera fetches a JPEG every ``PULL_INTERVAL`` seconds and hands it to the same
processing path as ``POST /cameras/{id}/frames``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from . import config


@dataclass
class Puller:
    camera_id: str
    url: str
    frames: int = 0
    errors: int = 0
    last_ok: Optional[float] = None
    last_error: Optional[str] = None
    last_fetch_ms: Optional[float] = None
    stop: threading.Event = field(default_factory=threading.Event, repr=False)

    def info(self) -> dict:
        return {"camera_id": self.camera_id, "url": self.url, "frames": self.frames, "errors": self.errors,
                "last_ok_age_s": None if self.last_ok is None else round(time.time() - self.last_ok, 1),
                "last_fetch_ms": self.last_fetch_ms, "last_error": self.last_error}


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


def _run(p: Puller, handle: Callable[[str, bytes], object]) -> None:
    sess = requests.Session()
    while not p.stop.is_set():
        t0 = time.time()
        try:
            r = sess.get(p.url, timeout=config.PULL_TIMEOUT)
            r.raise_for_status()
            p.last_fetch_ms = round((time.time() - t0) * 1000)
            handle(p.camera_id, r.content)
            p.frames += 1
            p.last_ok = time.time()
            if p.last_error:
                print(f"[bikeslot] pull {p.camera_id}: recovered")
            p.last_error = None
        except Exception as exc:  # camera offline, bad JPEG, ... - keep polling
            p.errors += 1
            msg = f"{type(exc).__name__}: {exc}"
            if msg != p.last_error:  # log each distinct error once, not every interval
                print(f"[bikeslot] pull {p.camera_id} from {p.url} failed: {msg}")
            p.last_error = msg
        p.stop.wait(max(0.0, config.PULL_INTERVAL - (time.time() - t0)))


def start(handle: Callable[[str, bytes], object], spec: str = None) -> None:
    for cam, url in parse(config.PULL_CAMERAS if spec is None else spec).items():
        if cam in pullers:
            continue
        p = pullers[cam] = Puller(cam, url)
        threading.Thread(target=_run, args=(p, handle), name=f"pull-{cam}", daemon=True).start()
        print(f"[bikeslot] pulling {cam} from {url} every {config.PULL_INTERVAL:g} s")


def stop_all() -> None:
    for p in pullers.values():
        p.stop.set()
    pullers.clear()


def info() -> list[dict]:
    return [p.info() for p in pullers.values()]
