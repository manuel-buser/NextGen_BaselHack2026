"""Settings from environment variables; the defaults suit a local demo."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    racks_file: str = "racks.yaml"
    detector_url: str = "fake"  # "fake" = built-in stand-in, otherwise e.g. http://localhost:8001
    detector_timeout_s: float = 15.0
    stale_after_s: float = 60.0  # no detection result for this long: the rack shows "stale"
    free_after_clean: int = 3  # pictures in a row without a bike before a slot shows free
    min_confidence: float = 0.4  # a "free" with less confidence than this counts as taken
    camera_pull: bool = True  # poll the racks that have a `camera` block; CAMERA_PULL=0 turns all polling off
    hot_limit_c: float = 75.0  # above this chip temperature a camera counts as too hot when the node does not say
    # None keeps the history in memory only (tests); HISTORY_FILE defaults to data/history.json, empty also means None
    history_file: str | None = None
    # Place search for the user page; GEOCODER_URL= (empty) turns it off
    geocoder_url: str = "https://nominatim.openstreetmap.org/search"
    # Racks with a `vision` block: how often the app asks the vision service for the decided spaces, and the shared
    # secret the service must send when it pushes them instead (X-Vision-Token); empty = no token needed
    vision_poll_s: float = 3.0
    vision_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            racks_file=env("BIKESLOT_RACKS", cls.racks_file),
            detector_url=env("DETECTOR_URL", cls.detector_url),
            detector_timeout_s=float(env("DETECTOR_TIMEOUT_S", cls.detector_timeout_s)),
            stale_after_s=float(env("STALE_AFTER_S", cls.stale_after_s)),
            free_after_clean=int(env("FREE_AFTER_CLEAN", cls.free_after_clean)),
            min_confidence=float(env("MIN_CONFIDENCE", cls.min_confidence)),
            camera_pull=env("CAMERA_PULL", "1") != "0",
            hot_limit_c=float(env("HOT_LIMIT_C", cls.hot_limit_c)),
            history_file=env("HISTORY_FILE", "data/history.json") or None,
            geocoder_url=env("GEOCODER_URL", cls.geocoder_url),
            vision_poll_s=float(env("VISION_POLL_S", cls.vision_poll_s)),
            vision_token=env("VISION_TOKEN", cls.vision_token),
        )
