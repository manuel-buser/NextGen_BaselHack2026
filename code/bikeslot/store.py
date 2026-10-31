"""In-memory camera state + on-disk layouts/history/previews.

Privacy: raw frames live only in a small RAM ring buffer per camera (for calibration); nothing raw is
ever written to disk.  Only layouts, per-slot status history and blurred previews are persisted.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .geometry import normalize, row_from_polygons, slots_from_rows
from .occupancy import SlotMaskCache, SlotTracker, make_trackers
from .schemas import (CameraInfo, CameraStatus, Detection, Layout, LayoutUpdate, SiteCamera, SiteSummary,
                      Slot, utcnow)


@dataclass
class CameraState:
    camera_id: str
    layout: Layout
    trackers: dict[str, SlotTracker] = field(default_factory=dict)
    cache: SlotMaskCache = field(default_factory=SlotMaskCache)
    frames: deque = field(default_factory=lambda: deque(maxlen=config.RING_BUFFER))  # (datetime, np.ndarray)
    last_status: Optional[CameraStatus] = None
    last_frame_at: Optional[datetime] = None
    last_detections: list[Detection] = field(default_factory=list)
    preview_jpeg: Optional[bytes] = None
    calibration_jpeg: Optional[bytes] = None

    def reset_trackers(self) -> None:
        self.trackers = make_trackers(self.layout)
        self.cache = SlotMaskCache()

    @property
    def online(self) -> bool:
        if self.last_frame_at is None:
            return False
        return (utcnow() - self.last_frame_at).total_seconds() < config.OFFLINE_AFTER_S


class Store:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.layout_dir = self.data_dir / "layouts"
        self.history_dir = self.data_dir / "history"
        self.preview_dir = self.data_dir / "preview"
        for d in (self.layout_dir, self.history_dir, self.preview_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.cameras: dict[str, CameraState] = {}
        self.lock = threading.RLock()
        self.load_layouts()

    # ------------------------------------------------------------------ cameras
    def get(self, camera_id: str) -> Optional[CameraState]:
        return self.cameras.get(camera_id)

    def get_or_create(self, camera_id: str) -> CameraState:
        with self.lock:
            cam = self.cameras.get(camera_id)
            if cam is None:
                cam = CameraState(camera_id=camera_id, layout=Layout(camera_id=camera_id))
                self.cameras[camera_id] = cam
            return cam

    def camera_infos(self) -> list[CameraInfo]:
        return [CameraInfo(camera_id=c.camera_id, site_id=c.layout.site_id, site_name=c.layout.site_name,
                           layout_status=c.layout.status, occupancy_mode=c.layout.occupancy_mode,
                           slot_count=len(c.layout.slots), online=c.online, last_frame_at=c.last_frame_at,
                           frames_buffered=len(c.frames))
                for c in self.cameras.values()]

    # ------------------------------------------------------------------ layouts
    def load_layouts(self) -> int:
        n = 0
        for path in sorted(self.layout_dir.glob("*.json")):
            try:
                layout = Layout.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:  # corrupt file -> skip, never crash the service
                print(f"[store] skipping {path.name}: {exc}")
                continue
            cam = self.get_or_create(layout.camera_id)
            cam.layout = layout
            cam.reset_trackers()
            n += 1
        return n

    def save_layout(self, layout: Layout) -> None:
        path = self.layout_dir / f"{layout.camera_id}.json"
        path.write_text(layout.model_dump_json(indent=2), encoding="utf-8")

    def set_layout(self, camera_id: str, layout: Layout, persist: bool = True) -> Layout:
        with self.lock:
            cam = self.get_or_create(camera_id)
            layout.camera_id = camera_id
            if layout.rows and not layout.slots:
                layout.slots = slots_from_rows(layout)
            layout.updated_at = utcnow()
            cam.layout = layout
            cam.reset_trackers()
            if persist:
                self.save_layout(layout)
            return layout

    def update_layout(self, camera_id: str, upd: LayoutUpdate) -> Layout:
        with self.lock:
            cam = self.get_or_create(camera_id)
            data = cam.layout.model_dump()
            changes = upd.model_dump(exclude_none=True)
            rows_changed = "rows" in changes
            slots_given = "slots" in changes
            data.update(changes)
            layout = Layout.model_validate(data)
            if rows_changed and not slots_given:
                layout.slots = slots_from_rows(layout)
            if not layout.rows and layout.slots:
                layout.rows = [row_from_polygons([s.polygon for s in layout.slots])]
            if layout.status == "none" and layout.slots:
                layout.status = "proposed"
            layout.source = "admin"
            return self.set_layout(camera_id, layout)

    def seed_golden(self, path: Path = config.GOLDEN_LABELS) -> Optional[Layout]:
        """Load the golden layout as a confirmed layout for its camera unless a saved layout exists."""
        if not path.exists():
            return None
        layout = layout_from_golden(path)
        if (self.layout_dir / f"{layout.camera_id}.json").exists():
            return self.get(layout.camera_id).layout
        return self.set_layout(layout.camera_id, layout, persist=False)

    # ------------------------------------------------------------------ frames / status
    def push_frame(self, cam: CameraState, frame: np.ndarray) -> None:
        cam.frames.append((utcnow(), frame))
        cam.last_frame_at = utcnow()

    def record_status(self, cam: CameraState, status: CameraStatus, dets: list[Detection]) -> None:
        cam.last_status = status
        cam.last_detections = dets
        line = {"ts": status.updated_at.isoformat() if status.updated_at else None, "camera_id": cam.camera_id,
                "free": status.free, "occupied": status.occupied, "unknown": status.unknown,
                "estimated": status.estimated, "slots": {s.slot_id: s.status for s in status.slots},
                "detections": status.detections, "processing_ms": status.processing_ms}
        with open(self.history_dir / f"{cam.camera_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def current_status(self, cam: CameraState) -> CameraStatus:
        """Latest status with freshness fields recomputed."""
        if cam.last_status is None:
            return CameraStatus(camera_id=cam.camera_id, site_id=cam.layout.site_id, online=cam.online,
                               layout_status=cam.layout.status, occupancy_mode=cam.layout.occupancy_mode,
                               slot_count=len(cam.layout.slots), unknown=len(cam.layout.slots),
                               message="no frame received yet")
        st = cam.last_status.model_copy()
        st.online = cam.online
        st.frame_age_s = round((utcnow() - cam.last_frame_at).total_seconds(), 1) if cam.last_frame_at else None
        st.preview_url = f"/cameras/{cam.camera_id}/preview.jpg" if cam.preview_jpeg else None
        if not st.online:
            st.message = (st.message + " | " if st.message else "") + "camera offline (no recent frame)"
        return st

    def sites(self) -> list[SiteSummary]:
        by_site: dict[str, SiteSummary] = {}
        for cam in self.cameras.values():
            st = self.current_status(cam)
            site = by_site.setdefault(cam.layout.site_id,
                                      SiteSummary(site_id=cam.layout.site_id, site_name=cam.layout.site_name))
            if not site.site_name and cam.layout.site_name:
                site.site_name = cam.layout.site_name
            site.slot_count += st.slot_count
            site.free += st.free
            site.occupied += st.occupied
            site.unknown += st.unknown
            site.estimated = site.estimated or st.estimated
            site.free_slot_ids.extend(s.slot_id for s in st.slots if s.status == "free")
            site.cameras.append(SiteCamera(camera_id=cam.camera_id, slot_count=st.slot_count, free=st.free,
                                           occupied=st.occupied, unknown=st.unknown, online=st.online,
                                           layout_status=st.layout_status, estimated=st.estimated))
            if st.updated_at and (site.updated_at is None or st.updated_at > site.updated_at):
                site.updated_at = st.updated_at
        return list(by_site.values())


# ---------------------------------------------------------------------- golden adapter

def layout_from_golden(path: Path = config.GOLDEN_LABELS, camera_index: int = 0) -> Layout:
    """Convert a camera entry of golden_labels.json into a confirmed Layout (normalised coordinates)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cam = data["cameras"][camera_index]
    frame_size = [int(cam["image_size"][0]), int(cam["image_size"][1])]
    slots = [Slot(slot_id=s["slot_id"], index=int(s["index"]), row_id="r1",
                  polygon=normalize(s["polygon"], frame_size))
             for s in sorted(cam["slots"], key=lambda s: s["index"])]
    row = row_from_polygons([s.polygon for s in slots])
    return Layout(camera_id=cam["camera_id"], site_id=cam.get("location_id", "site01"),
                  site_name=cam.get("location_name", ""), frame_size=frame_size,
                  parking_type="painted_bays", status="confirmed", source="golden",
                  occupancy_mode="per_slot", rows=[row], slots=slots)


def golden_truth(path: Path = config.GOLDEN_LABELS) -> list[dict]:
    """[{file, camera_id, slots: {slot_id: 'free'|'occupied'}}] for evaluation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"file": im["file"], "camera_id": im["camera_id"], "slots": im["slots"], "notes": im.get("notes", "")}
            for im in data["images"]]
