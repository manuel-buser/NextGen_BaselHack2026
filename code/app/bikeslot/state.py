"""In-memory state: racks, slots, camera nodes, detector health. Pictures are never kept."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

import yaml

from .config import Settings
from .history import History

TAKEN, FREE = "taken", "free"


class Slot:
    """Remembers the last few observations of one parking space (True = a bike was seen)."""

    def __init__(self, window: int) -> None:
        self.recent: deque[bool] = deque(maxlen=window)
        self.confidence = 0.0

    def observe(self, bike_seen: bool, confidence: float) -> None:
        self.recent.append(bike_seen)
        self.confidence = confidence

    @property
    def ready(self) -> bool:
        return len(self.recent) == self.recent.maxlen

    @property
    def state(self) -> str:
        # One sighting shows taken at once; free needs a full window of clean pictures (safe side).
        if any(self.recent):
            return TAKEN
        return FREE if self.ready else "unknown"


@dataclass
class Camera:
    node_id: str | None = None
    source: str | None = None  # "push": the node posts to the app; "pull": the app polls the node; "vision": the vision service decides (vision.py)
    last_alive_at: float | None = None
    last_picture_at: float | None = None
    pictures: int = 0
    last_picture_bytes: int = 0
    temp_c: float | None = None
    too_hot: bool = False
    too_hot_derived: bool = False  # the node sent no flag, so too_hot is temp_c compared with HOT_LIMIT_C
    last_error: str | None = None
    last_error_at: float | None = None
    # What the node reports about itself (pull status JSON, or the same keys in a pushed alive message).
    uptime_s: int | None = None
    rebooted_at: float | None = None  # when the uptime last dropped below the previous reading
    wifi_rssi_dbm: int | None = None
    wifi_connected: bool | None = None
    wifi_channel: int | None = None
    heap_free: int | None = None
    heap_min_free: int | None = None
    heap_largest_block: int | None = None
    psram_free: int | None = None
    psram_total: int | None = None
    stream_frames: int | None = None
    stream_dropped: int | None = None
    ip: str | None = None


NODE_FIELDS = (
    "uptime_s", "rebooted_at", "wifi_rssi_dbm", "wifi_connected", "wifi_channel", "heap_free", "heap_min_free",
    "heap_largest_block", "psram_free", "psram_total", "stream_frames", "stream_dropped", "ip",
)


@dataclass(frozen=True)
class PullCamera:
    """The `camera` block of a rack: where the app polls the node instead of waiting for its posts."""

    url: str
    status_path: str = "/status"
    picture_path: str = "/capture"
    poll_s: float = 5.0
    picture_every_s: float = 10.0

    @property
    def node_id(self) -> str:
        return urlsplit(self.url).hostname or self.url

    @property
    def status_url(self) -> str:
        return self.url + self.status_path

    @property
    def picture_url(self) -> str:
        return self.url + self.picture_path


def camera_from_yaml(raw: dict | None, rack_id: str) -> PullCamera | None:
    if raw is None:
        return None
    url = str(raw.get("url", "")).rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"rack {rack_id}: camera url must start with http:// or https://, got {url!r}")
    defaults = PullCamera(url)
    status_path, picture_path = str(raw.get("status_path", defaults.status_path)), str(raw.get("picture_path", defaults.picture_path))
    if not status_path.startswith("/") or not picture_path.startswith("/"):
        raise ValueError(f"rack {rack_id}: camera status_path and picture_path must start with /")
    poll_s, picture_every_s = float(raw.get("poll_s", defaults.poll_s)), float(raw.get("picture_every_s", defaults.picture_every_s))
    if poll_s <= 0 or picture_every_s <= 0:
        raise ValueError(f"rack {rack_id}: camera poll_s and picture_every_s must be above 0")
    return PullCamera(url, status_path, picture_path, poll_s, picture_every_s)


@dataclass(frozen=True)
class VisionSource:
    """The `vision` block of a rack: the recognition service on the vision laptop keeps this rack's camera and
    decides its spaces itself; the app only reads the result (GET {url}/cameras/{camera}/status every poll_s), or
    the service pushes it (POST /api/vision/{rack}/status) when `url` is left out. See bikeslot/vision.py."""

    camera: str
    url: str | None = None
    poll_s: float = 3.0

    @property
    def node_id(self) -> str:
        return (urlsplit(self.url).hostname if self.url else None) or self.camera

    @property
    def status_url(self) -> str:
        return f"{self.url}/cameras/{self.camera}/status"

    @property
    def preview_url(self) -> str:
        return f"{self.url}/cameras/{self.camera}/preview.jpg"


def vision_from_yaml(raw: dict | None, rack_id: str, default_poll_s: float) -> VisionSource | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw.get("camera"):
        raise ValueError(f"rack {rack_id}: vision needs the camera id used by the vision service, e.g. vision: {{camera: live01, url: 'http://192.168.21.88:8000'}}")
    url = raw.get("url")
    if url is not None:
        url = str(url).rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"rack {rack_id}: vision url must start with http:// or https://, got {url!r}")
    poll_s = float(raw.get("poll_s", default_poll_s))
    if poll_s <= 0:
        raise ValueError(f"rack {rack_id}: vision poll_s must be above 0")
    return VisionSource(str(raw["camera"]), url, poll_s)


@dataclass
class DetectorHealth:
    last_ok_at: float | None = None
    last_latency_ms: float | None = None
    model: str | None = None
    last_error: str | None = None
    last_error_at: float | None = None


def _num(value) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def layout_from_yaml(raw: dict | None, capacity: int, rack_id: str) -> dict | None:
    """Turn the optional `layout` of a rack into one shape in grid units: x runs along the bearing from
    space 1, y across it (row 1 is y 0). Either `rows`/`cols` or an explicit `spaces` list."""
    if raw is None:
        return None
    bearing = _num(raw.get("bearing", 0))
    entrance = str(raw.get("entrance", "start"))
    if not 0 <= bearing <= 360:
        raise ValueError(f"rack {rack_id}: layout bearing must be 0 to 360 degrees, got {bearing}")
    if entrance not in ("start", "end"):
        raise ValueError(f"rack {rack_id}: layout entrance must be 'start' or 'end', got {entrance!r}")
    space_m = raw.get("space_m", [0.7, 1.9])
    if not isinstance(space_m, list) or len(space_m) != 2 or not all(_num(v) > 0 for v in space_m):
        raise ValueError(f"rack {rack_id}: layout space_m must be [along_row, depth] in metres, both above 0, got {space_m!r}")
    row_gap_m = _num(raw.get("row_gap_m", 0.6))
    if row_gap_m < 0:
        raise ValueError(f"rack {rack_id}: layout row_gap_m must be 0 or more metres, got {row_gap_m}")
    if "spaces" in raw:
        spaces = [
            {"id": int(s["id"]), "x": _num(s["x"]), "y": _num(s["y"]), "w": _num(s.get("w", 1)), "h": _num(s.get("h", 1))}
            for s in raw["spaces"]
        ]
    else:
        rows, cols = int(raw["rows"]), int(raw["cols"])
        spaces = [{"id": r * cols + c + 1, "x": c, "y": r, "w": 1, "h": 1} for r in range(rows) for c in range(cols)]
    if sorted(s["id"] for s in spaces) != list(range(1, capacity + 1)):
        raise ValueError(f"rack {rack_id}: layout space ids must be exactly 1 to {capacity} (the capacity)")
    return {
        "bearing": bearing,
        "entrance": entrance,
        "space_m": [_num(v) for v in space_m],
        "row_gap_m": row_gap_m,
        "width": max(s["x"] + s["w"] for s in spaces),
        "height": max(s["y"] + s["h"] for s in spaces),
        "spaces": sorted(spaces, key=lambda s: s["id"]),
    }


class Rack:
    def __init__(
        self,
        rack_id: str,
        name: str,
        capacity: int,
        window: int,
        location: dict | None = None,
        demo: bool = False,
        layout: dict | None = None,
        pull: PullCamera | None = None,
        vision: VisionSource | None = None,
        group: str | None = None,
    ) -> None:
        self.id = rack_id
        self.name = name
        self.capacity = capacity
        self.window = window
        self.location = location  # {"lat": .., "lng": ..} or None: list only, no map pin
        self.demo = demo  # made up, fed by tools/demo_data.py: labelled on the pages, never sent to the real detector
        self.layout = layout  # drawn plan of the spaces (see layout_from_yaml) or None: plain square row
        self.pull = pull  # the app polls this node (see camera_from_yaml) or None: the node posts to the app
        self.vision = vision  # the vision service keeps this rack's camera and decides its spaces (see vision_from_yaml)
        self.vision_seen: str | None = None  # updated_at of the last status taken from the service: a new one is a new frame
        self.vision_state: dict | None = None  # the service's own view of the rack, shown on the admin page
        self.tracking = True  # False while the vision service has no camera picture: all spaces count as taken, shown grey
        self.group = group  # the location this rack is a part of on the user page, or None: shown on its own
        self.slots: dict[int, Slot] = {}
        self.counts: deque[int] = deque(maxlen=window)  # count mode: free spaces observed
        self.mode: str | None = None  # "slots" or "count", set by the first detector result
        self.camera = Camera()
        self.last_result_at: float | None = None

    def apply(self, result: dict, settings: Settings, now: float) -> None:
        slots = result.get("slots")
        if slots:
            self.mode = "slots"
            for item in slots:
                confidence = float(item.get("confidence", 1.0))
                clean = item.get("state") == FREE and confidence >= settings.min_confidence
                self.slots.setdefault(int(item["id"]), Slot(self.window)).observe(not clean, confidence)
        elif "bike_count" in result:
            self.mode = "count"
            self.counts.append(max(0, self.capacity - int(result["bike_count"])))
        else:
            raise ValueError("neither slots nor bike_count")
        self.last_result_at = now

    @property
    def ready(self) -> bool:
        if self.mode == "slots":
            return bool(self.slots) and all(slot.ready for slot in self.slots.values())
        if self.mode == "count":
            return len(self.counts) == self.counts.maxlen
        return False

    @property
    def free(self) -> int | None:
        if not self.ready:
            return None
        if self.mode == "slots":
            return sum(1 for slot in self.slots.values() if slot.state == FREE)
        return min(self.counts)  # drops at once, rises only when the whole window agrees

    def vision_view(self) -> dict | None:
        if self.vision is None:
            return None
        return {"camera": self.vision.camera, "url": self.vision.url, "how": "pull" if self.vision.url else "push",
                "poll_s": self.vision.poll_s, **(self.vision_state or {})}

    def status(self, settings: Settings, now: float) -> str:
        if self.last_result_at is None:
            return "unknown"
        if not self.ready:
            return "starting"
        if now - self.last_result_at > settings.stale_after_s:
            return "stale"
        return "ok"


class Location:
    """The racks with the same `group` in the racks file: one place, so the user page shows them as one entry."""

    def __init__(self, name: str, parts: list[Rack]) -> None:
        self.id = self.name = name
        self.parts = parts
        self.capacity = sum(part.capacity for part in parts)
        placed = [part.location for part in parts if part.location]
        self.location = {key: sum(p[key] for p in placed) / len(placed) for key in ("lat", "lng")} if placed else None

    def status(self, settings: Settings, now: float) -> str:
        states = {part.status(settings, now) for part in self.parts}
        return next((state for state in ("ok", "starting", "stale") if state in states), "unknown")


class Store:
    def __init__(self, racks: list[Rack], settings: Settings, places: list[dict] | None = None) -> None:
        self.racks = {rack.id: rack for rack in racks}
        self.places = places or []  # destinations people go to: {"name", "location"}
        self.settings = settings
        self.detector = DetectorHealth()
        self.started_at = time.time()
        self.history = History(settings.history_file)
        groups: dict[str, list[Rack]] = {}
        for rack in racks:
            if rack.group:
                groups.setdefault(rack.group, []).append(rack)
        self.locations = {name: Location(name, parts) for name, parts in groups.items()}
        self.history.groups = {name: [part.id for part in parts] for name, parts in groups.items()}

    @classmethod
    def from_yaml(cls, path: str, settings: Settings) -> "Store":
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        entries = raw["racks"]
        places = []
        for p in raw.get("places") or []:
            if not isinstance(p, dict) or not p.get("name") or not isinstance(p.get("location"), dict):
                raise ValueError(f"places: each place needs a name and a location {{lat, lng}}, got {p!r}")
            lat, lng = float(p["location"]["lat"]), float(p["location"]["lng"])
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                raise ValueError(f"places: {p['name']}: location is off the globe ({lat}, {lng})")
            places.append({"name": str(p["name"]), "location": {"lat": lat, "lng": lng}})
        racks = []
        for e in entries:
            loc = e.get("location")
            location = {"lat": float(loc["lat"]), "lng": float(loc["lng"])} if loc else None
            demo = bool(e.get("demo", False))
            rack_id, capacity = str(e["id"]), int(e["capacity"])
            layout = layout_from_yaml(e.get("layout"), capacity, rack_id)
            pull = camera_from_yaml(e.get("camera"), rack_id)
            group = str(e["group"]) if e.get("group") else None
            vision = vision_from_yaml(e.get("vision"), rack_id, settings.vision_poll_s)
            if vision and (pull or demo):
                raise ValueError(f"rack {rack_id}: vision cannot be combined with camera or demo")
            window = 1 if vision else settings.free_after_clean  # the vision service has smoothed its states already
            racks.append(Rack(rack_id, str(e["name"]), capacity, window, location, demo, layout, pull, vision, group))
        clashes = {rack.group for rack in racks} & {rack.id for rack in racks}
        if clashes:
            raise ValueError(f"group {sorted(clashes)[0]!r} is also a rack id; the user page needs one name per entry")
        return cls(racks, settings, places)

    def note_alive(self, rack: Rack, report: dict, node_id: str | None, now: float, source: str) -> None:
        """Record a sign of life; `report` is the node's data normalised by ingest.node_report (Camera field names)."""
        cam = rack.camera
        cam.last_alive_at = now
        cam.node_id = node_id or cam.node_id
        cam.source = source
        if cam.uptime_s is not None and report.get("uptime_s") is not None and report["uptime_s"] < cam.uptime_s:
            cam.rebooted_at = now  # the counter started again, so the node restarted
        for key, value in report.items():
            setattr(cam, key, value)

    def note_picture(self, rack: Rack, nbytes: int, node_id: str | None, now: float, source: str) -> None:
        cam = rack.camera
        cam.last_picture_at = now
        cam.last_alive_at = now
        cam.pictures += 1
        cam.last_picture_bytes = nbytes
        cam.node_id = node_id or cam.node_id
        cam.source = source

    def note_camera_error(self, rack: Rack, message: str, now: float) -> None:
        rack.camera.last_error = message
        rack.camera.last_error_at = now

    def apply_detection(self, rack: Rack, result: dict, latency_ms: float, now: float, source: str = "detector") -> None:
        rack.apply(result, self.settings, now)
        if rack.status(self.settings, now) == "ok":
            self.history.record(rack.id, datetime.fromtimestamp(now), rack.free, rack.capacity)
        if rack.demo or source == "vision":  # the stand-in or the vision service answered: nothing about the detector
            return
        self.detector.last_ok_at = now
        self.detector.last_latency_ms = round(latency_ms, 1)
        self.detector.model = result.get("model")

    def note_detector_error(self, message: str, now: float) -> None:
        self.detector.last_error = message
        self.detector.last_error_at = now

    def free_now(self, rack: Rack | Location, now: float) -> int | None:
        """Free spaces with live data; a location adds up its live racks (one without live data adds none)."""
        if isinstance(rack, Location):
            live = [free for free in (self.free_now(part, now) for part in rack.parts) if free is not None]
            return sum(live) if live else None
        return rack.free if rack.status(self.settings, now) == "ok" else None

    def public_view(self, now: float) -> list[dict]:
        """What the user page lists: each rack on its own, or its location in the place of the location's first rack."""
        views, shown = [], set()
        for rack in self.racks.values():
            if rack.group is None:
                views.append(self.rack_view(rack, now))
            elif rack.group not in shown:
                shown.add(rack.group)
                views.append(self.location_view(self.locations[rack.group], now))
        return views

    def location_view(self, loc: Location, now: float) -> dict:
        parts = [self.rack_view(part, now, forecast=False) for part in loc.parts]
        seen = [part.last_result_at for part in loc.parts if part.last_result_at is not None]
        return {
            "id": loc.id,
            "name": loc.name,
            "capacity": loc.capacity,
            "free": self.free_now(loc, now),
            "status": loc.status(self.settings, now),
            "mode": None,
            "updated_at": max(seen) if seen else None,
            "age_s": round(now - max(seen), 1) if seen else None,
            "camera_too_hot": any(part.camera.too_hot for part in loc.parts),
            "location": loc.location,
            "demo": all(part.demo for part in loc.parts),
            "layout": None,
            "slots": [],
            "tracking": all(part.tracking for part in loc.parts),
            "forecast": self.history.forecast(loc, now),
            "parts": parts,
        }

    def rack_view(self, rack: Rack, now: float, forecast: bool = True) -> dict:
        view = {
            "id": rack.id,
            "name": rack.name,
            "capacity": rack.capacity,
            "free": rack.free,
            "status": rack.status(self.settings, now),
            "mode": rack.mode,
            "updated_at": rack.last_result_at,
            "age_s": None if rack.last_result_at is None else round(now - rack.last_result_at, 1),
            "camera_too_hot": rack.camera.too_hot,
            "location": rack.location,
            "demo": rack.demo,
            "source": "demo" if rack.demo else "vision" if rack.vision else "camera",
            "layout": rack.layout,
            "slots": [{"id": sid, "state": slot.state} for sid, slot in sorted(rack.slots.items())],
            "tracking": rack.tracking,
        }
        if forecast:
            view["forecast"] = self.history.forecast(rack, now)
        return view

    def history_view(self, rack: Rack | Location, now: float) -> dict:
        return self.history.view(rack, now, self.free_now(rack, now))

    def admin_view(self, now: float) -> dict:
        def age(stamp: float | None) -> float | None:
            return None if stamp is None else round(now - stamp, 1)

        cameras = [
            {
                "rack": rack.id,
                "demo": rack.demo,
                "node_id": rack.camera.node_id,
                "source": rack.camera.source,
                "last_alive_at": rack.camera.last_alive_at,
                "alive_age_s": age(rack.camera.last_alive_at),
                "last_picture_at": rack.camera.last_picture_at,
                "picture_age_s": age(rack.camera.last_picture_at),
                "pictures": rack.camera.pictures,
                "last_picture_bytes": rack.camera.last_picture_bytes,
                "temp_c": rack.camera.temp_c,
                "too_hot": rack.camera.too_hot,
                "too_hot_derived": rack.camera.too_hot_derived,
                "last_error": rack.camera.last_error,
                "last_error_at": rack.camera.last_error_at,
                "error_age_s": age(rack.camera.last_error_at),
                "online": rack.camera.last_alive_at is not None
                and now - rack.camera.last_alive_at <= self.settings.stale_after_s
                and rack.tracking,  # a vision rack whose service has no camera picture: the camera is offline
                "node": {key: getattr(rack.camera, key) for key in NODE_FIELDS},
                "vision": rack.vision_view(),
            }
            for rack in self.racks.values()
        ]
        racks = [
            {
                **self.rack_view(rack, now),
                "slots_observed": len(rack.slots),
                "slots_ready": sum(1 for slot in rack.slots.values() if slot.ready),
            }
            for rack in self.racks.values()
        ]
        return {
            "generated_at": now,
            "app": {
                "started_at": self.started_at,
                "uptime_s": round(now - self.started_at, 1),
                "racks_file": self.settings.racks_file,
                "stale_after_s": self.settings.stale_after_s,
                "free_after_clean": self.settings.free_after_clean,
                "min_confidence": self.settings.min_confidence,
            },
            "detector_results": {
                "last_ok_at": self.detector.last_ok_at,
                "ok_age_s": age(self.detector.last_ok_at),
                "last_latency_ms": self.detector.last_latency_ms,
                "model": self.detector.model,
                "last_error": self.detector.last_error,
                "last_error_at": self.detector.last_error_at,
                "error_age_s": age(self.detector.last_error_at),
            },
            "cameras": cameras,
            "racks": racks,
        }
