"""Per frame: slot polygons -> crops -> classifier -> smoothing -> CameraStatus (same schema as Stage B).

    analyze_frame(layout, img, camera_id, trackers)   # one call per snapshot

The only model that runs is the small crop classifier.  Optionally the YOLO detector is run too, purely to
blur persons in the previews and to mark slots hidden behind a person as not observable (privacy rule of
the project; switch off with BIKESLOT_SLOTCLS_BLUR_PERSONS=0 on a device without GPU).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import config
from ..geometry import denormalize, laplacian_var, resize_max_side, slot_masks
from ..occupancy import SlotTracker
from ..schemas import CameraStatus, Detection, Layout, SlotStatus, utcnow
from . import smoothing
from .crops import slot_crops
from .model import get_model


@dataclass
class ClassifierResult:
    status: CameraStatus
    frame: np.ndarray                      # working-resolution BGR frame (raw, NOT blurred)
    probs: dict[str, float]                # slot_id -> P(occupied) in this frame
    crops: dict[str, np.ndarray]
    detections: list[Detection] = field(default_factory=list)   # persons (for blurring), may be empty
    assignments: dict = field(default_factory=dict)             # kept for preview.render compatibility


def _pixels(poly, frame_size):
    return [[round(x, 1), round(y, 1)] for x, y in denormalize(poly, frame_size)]


def person_detections(frame: np.ndarray) -> list[Detection]:
    from .. import detector

    try:
        return [d for d in detector.detect(frame) if d.cls == config.CLS_PERSON]
    except Exception as exc:  # detector weights missing / no GPU: never block the classifier
        print(f"[slotcls] person detector unavailable: {exc}")
        return []


def analyze_frame(layout: Layout, img_bgr: np.ndarray, camera_id: str,
                  trackers: Optional[dict[str, SlotTracker]] = None, model=None,
                  blur_persons: bool = config.SLOTCLS_BLUR_PERSONS, model_path=None) -> ClassifierResult:
    """Classify every slot of ``layout`` in the full-resolution frame ``img_bgr``."""
    t0 = time.time()
    frame, _ = resize_max_side(img_bgr, config.MAX_SIDE)
    h, w = frame.shape[:2]
    frame_size = (w, h)
    p = layout.params or {}
    margin = float(p.get("crop_margin", config.SLOTCLS_MARGIN))
    unsure = float(p.get("unsure_band", config.SLOTCLS_UNSURE_BAND))
    status = CameraStatus(camera_id=camera_id, site_id=layout.site_id, online=True, frame_age_s=0.0,
                          updated_at=utcnow(), layout_status=layout.status, occupancy_mode="per_slot",
                          frame_size=[w, h], slot_count=len(layout.slots))
    if trackers is None:
        trackers = smoothing.make_trackers(layout)

    if layout.status == "none" or not layout.slots:
        status.message = "no slot polygons for this camera yet - calibrate (slotcls_calibrate.py or PUT /cameras/{id}/layout)"
        status.processing_ms = round((time.time() - t0) * 1000, 1)
        return ClassifierResult(status, frame, {}, {})

    persons = person_detections(frame) if blur_persons else []
    status.detections = {"person": len(persons), "bicycle": 0, "motorcycle": 0}

    model = model or get_model(model_path or config.SLOTCLS_MODEL)
    crops = slot_crops(layout, img_bgr, size=model.input_size, margin=margin)
    ids = list(crops)
    probs = {sid: float(pr) for sid, pr in zip(ids, model.predict_proba([crops[s] for s in ids]))}

    votes: dict[str, Optional[float]] = dict(probs)
    if laplacian_var(frame) < config.BLUR_VAR_MIN:
        status.message = "frame too blurry - no votes cast"
        votes = {sid: None for sid in ids}
    if persons:
        masks = slot_masks(layout, frame_size)
        for person in persons:
            x1, y1, x2, y2 = [int(round(v)) for v in person.box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            for sid, (mask, _) in masks.items():
                area = mask.sum()
                if area and mask[y1:y2, x1:x2].sum() / area > config.PERSON_COVER_UNKNOWN:
                    votes[sid] = None
    smoothing.update(trackers, votes, unsure)

    status.slots = []
    for s in layout.slots:
        tr = trackers.setdefault(s.slot_id, SlotTracker(s.slot_id))
        status.slots.append(SlotStatus(slot_id=s.slot_id, index=s.index, status=tr.state, confidence=tr.confidence,
                                       score=round(probs.get(s.slot_id, 0.0), 3), last_change=tr.last_change,
                                       polygon=_pixels(s.polygon, frame_size)))
    status.free = sum(1 for x in status.slots if x.status == "free")
    status.occupied = sum(1 for x in status.slots if x.status == "occupied")
    status.unknown = sum(1 for x in status.slots if x.status == "unknown")
    status.processing_ms = round((time.time() - t0) * 1000, 1)
    return ClassifierResult(status, frame, probs, crops, persons)


def info() -> dict:
    from . import model as m

    return m.info()
