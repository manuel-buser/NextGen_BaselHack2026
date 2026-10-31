"""Stage B: one camera frame -> per-slot occupancy.

    detections (masks + ground-contact points)  ->  slot scores  ->  per-slot vote  ->  smoothing  ->  CameraStatus

Scoring a vehicle detection against a slot polygon:
    score = 1.0 if its ground-contact point (lowest mask pixels ~ rear wheel) lies inside the slot
          + share of its "on the ground" mask pixels (bottom FOOTPRINT_FRAC of the box) inside the slot
Each detection is assigned to its best slot only (one bike cannot fill two slots); a slot is voted
"occupied" in this frame if at least one vehicle was assigned to it.  Slots largely covered by a person
get no vote (unknown) because we cannot see them.  Votes are smoothed over the last k frames with
hysteresis so a single missed detection does not flip the state.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config, detector
from .geometry import laplacian_var, point_in_polygon, resize_max_side, row_hull, slot_masks
from .schemas import CameraStatus, Detection, Layout, SlotStatus, utcnow


@dataclass
class SlotTracker:
    """Temporal smoothing for one slot."""
    slot_id: str
    k: int = config.K_FRAMES
    min_frames: int = 2
    votes: deque = field(default_factory=deque)
    state: str = "unknown"
    last_change = None

    def __post_init__(self):
        self.votes = deque(maxlen=self.k)

    def update(self, vote: Optional[float]) -> None:
        """vote: 1.0 occupied, 0.0 free, None = slot not observable in this frame."""
        if vote is not None:
            self.votes.append(float(vote))
        mean = self.mean
        new_state = self.state
        if vote is None and self.state == "unknown":
            new_state = "unknown"
        elif len(self.votes) < self.min_frames:
            new_state = "unknown"
        elif mean >= config.OCCUPIED_ABOVE:
            new_state = "occupied"
        elif mean <= config.FREE_BELOW:
            new_state = "free"
        elif self.state == "unknown":
            new_state = "occupied" if mean > 0.5 else "free"
        if new_state != self.state:
            self.state = new_state
            self.last_change = utcnow()

    @property
    def mean(self) -> float:
        return float(sum(self.votes) / len(self.votes)) if self.votes else 0.0

    @property
    def confidence(self) -> float:
        if not self.votes or self.state == "unknown":
            return 0.0
        return round(min(1.0, abs(self.mean - 0.5) * 2.0 * min(1.0, len(self.votes) / self.min_frames)), 3)


@dataclass
class FrameResult:
    status: CameraStatus
    frame: np.ndarray                 # working-resolution BGR frame (raw, NOT blurred)
    detections: list[Detection]
    assignments: dict[int, str]       # detection index -> slot_id


class SlotMaskCache:
    """Rasterised slot masks for a (layout version, frame size) pair."""

    def __init__(self):
        self._key = None
        self.masks: dict = {}
        self.hull = None

    def get(self, layout: Layout, frame_size: tuple[int, int]):
        key = (layout.updated_at, tuple(frame_size), len(layout.slots))
        if key != self._key:
            self.masks = slot_masks(layout, frame_size)
            self.hull = row_hull(layout, frame_size)
            self._key = key
        return self.masks, self.hull


def make_trackers(layout: Layout) -> dict[str, SlotTracker]:
    p = layout.params or {}
    k = int(p.get("k_frames", config.K_FRAMES))
    min_frames = int(p.get("min_frames", 2))
    return {s.slot_id: SlotTracker(s.slot_id, k=k, min_frames=min_frames) for s in layout.slots}


def score_detection(det: Detection, masks: dict, shape: tuple[int, int], footprint_frac: float) -> dict[str, float]:
    """Score one vehicle detection against every slot (see module docstring)."""
    scores: dict[str, float] = {}
    ys, xs = detector.mask_pixels(det, shape)
    bottom = ys >= det.box[3] - footprint_frac * det.height if len(ys) else ys
    ys_b, xs_b = ys[bottom], xs[bottom]
    contact = det.contact or det.bottom_center
    for slot_id, (mask, poly) in masks.items():
        s = 1.0 if point_in_polygon(poly, contact) else 0.0
        if len(ys_b):
            s += float(mask[ys_b, xs_b].mean())
        if s > 0.0:
            scores[slot_id] = s
    return scores


def analyze_frame(layout: Layout, trackers: dict[str, SlotTracker], cache: SlotMaskCache,
                  img_bgr: np.ndarray, camera_id: str) -> FrameResult:
    """Run the full Stage-B pipeline on one raw frame."""
    t0 = time.time()
    frame, _ = resize_max_side(img_bgr, config.MAX_SIDE)
    h, w = frame.shape[:2]
    frame_size = (w, h)
    p = layout.params or {}
    min_assign = float(p.get("min_assign_score", config.MIN_ASSIGN_SCORE))
    footprint_frac = float(p.get("footprint_frac", config.FOOTPRINT_FRAC))

    status = CameraStatus(camera_id=camera_id, site_id=layout.site_id, online=True, frame_age_s=0.0,
                          updated_at=utcnow(), layout_status=layout.status,
                          occupancy_mode=layout.occupancy_mode, frame_size=[w, h],
                          slot_count=len(layout.slots))
    sharpness = laplacian_var(frame)
    dets = detector.detect(frame)
    counts: dict[str, int] = {name: 0 for name in config.CLASS_NAMES.values()}
    for d in dets:
        counts[d.name] = counts.get(d.name, 0) + 1
    status.detections = counts
    assignments: dict[int, str] = {}

    if sharpness < config.BLUR_VAR_MIN:
        status.message = f"frame too blurry (sharpness {sharpness:.0f} < {config.BLUR_VAR_MIN:.0f})"
        _fill_unknown(status, layout, trackers, frame_size)
        status.processing_ms = round((time.time() - t0) * 1000, 1)
        return FrameResult(status, frame, dets, assignments)

    if layout.status == "none" or not layout.slots:
        status.message = "no layout for this camera yet - POST /cameras/{id}/calibrate, then PUT /cameras/{id}/layout"
        status.processing_ms = round((time.time() - t0) * 1000, 1)
        return FrameResult(status, frame, dets, assignments)

    masks, hull = cache.get(layout, frame_size)
    vehicles = [(i, d) for i, d in enumerate(dets) if d.cls in config.VEHICLE_CLASSES]
    persons = [d for d in dets if d.cls == config.CLS_PERSON]

    if layout.occupancy_mode == "count_only":
        inside = 0
        for i, d in vehicles:
            if hull is not None and point_in_polygon(hull, d.contact or d.bottom_center):
                inside += 1
                assignments[i] = "row"
        n = len(layout.slots)
        status.occupied = min(n, inside)
        status.free = max(0, n - status.occupied)
        status.unknown = 0
        status.estimated = True
        status.slots = [SlotStatus(slot_id=s.slot_id, index=s.index, status="unknown",
                                   polygon=_pixels(s.polygon, frame_size)) for s in layout.slots]
        status.message = "count_only mode: per-slot state not available, counts are estimates"
        status.processing_ms = round((time.time() - t0) * 1000, 1)
        return FrameResult(status, frame, dets, assignments)

    # --- per-slot mode -----------------------------------------------------------------------
    votes: dict[str, Optional[float]] = {sid: 0.0 for sid in masks}
    for i, d in vehicles:
        scores = score_detection(d, masks, (h, w), footprint_frac)
        if not scores:
            continue
        best = max(scores, key=scores.get)
        if scores[best] >= min_assign:
            votes[best] = 1.0
            assignments[i] = best
    # slots mostly hidden behind a person are not observable in this frame
    for person in persons:
        x1, y1, x2, y2 = [int(round(v)) for v in person.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        for sid, (mask, _) in masks.items():
            area = mask.sum()
            if area and votes[sid] != 1.0 and mask[y1:y2, x1:x2].sum() / area > config.PERSON_COVER_UNKNOWN:
                votes[sid] = None

    status.slots = []
    for s in layout.slots:
        tr = trackers.setdefault(s.slot_id, SlotTracker(s.slot_id))
        tr.update(votes.get(s.slot_id, 0.0))
        status.slots.append(SlotStatus(slot_id=s.slot_id, index=s.index, status=tr.state,
                                       confidence=tr.confidence, score=round(tr.mean, 3),
                                       last_change=tr.last_change, polygon=_pixels(s.polygon, frame_size)))
    status.free = sum(1 for x in status.slots if x.status == "free")
    status.occupied = sum(1 for x in status.slots if x.status == "occupied")
    status.unknown = sum(1 for x in status.slots if x.status == "unknown")
    status.processing_ms = round((time.time() - t0) * 1000, 1)
    return FrameResult(status, frame, dets, assignments)


def _pixels(poly, frame_size):
    from .geometry import denormalize
    return [[round(x, 1), round(y, 1)] for x, y in denormalize(poly, frame_size)]


def _fill_unknown(status: CameraStatus, layout: Layout, trackers: dict[str, SlotTracker], frame_size) -> None:
    status.slots = []
    for s in layout.slots:
        tr = trackers.get(s.slot_id)
        state = tr.state if tr else "unknown"
        status.slots.append(SlotStatus(slot_id=s.slot_id, index=s.index, status=state,
                                       confidence=tr.confidence if tr else 0.0,
                                       score=round(tr.mean, 3) if tr else 0.0,
                                       last_change=tr.last_change if tr else None,
                                       polygon=_pixels(s.polygon, frame_size)))
    status.free = sum(1 for x in status.slots if x.status == "free")
    status.occupied = sum(1 for x in status.slots if x.status == "occupied")
    status.unknown = sum(1 for x in status.slots if x.status == "unknown")
