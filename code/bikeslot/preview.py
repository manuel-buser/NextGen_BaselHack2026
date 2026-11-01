"""Annotated, privacy-safe preview images (persons and motorcycles blurred BEFORE anything is drawn)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config
from .detector import blur_boxes
from .schemas import CameraStatus, Detection

COLORS = {"free": (80, 200, 80), "occupied": (60, 60, 230), "unknown": (160, 160, 160)}
DET_COLORS = {"bicycle": (0, 200, 0), "motorcycle": (0, 140, 255), "person": (0, 0, 255)}


def render(frame: np.ndarray, status: CameraStatus, dets: list[Detection], assignments: Optional[dict] = None,
           max_width: int = config.PREVIEW_MAX_WIDTH, show_detections: bool = True, show_scores: bool = False) -> bytes:
    """Return JPEG bytes of the blurred, annotated frame (slots coloured by status).

    ``show_scores`` prints each slot's score next to its number (the classifier engine's P(occupied))."""
    img = blur_boxes(frame, dets, inplace=False)
    overlay = img.copy()
    for s in status.slots:
        poly = np.array(s.polygon, dtype=np.float32).round().astype(np.int32)
        if len(poly) < 3:
            continue
        cv2.fillPoly(overlay, [poly], COLORS[s.status])
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    for s in status.slots:
        poly = np.array(s.polygon, dtype=np.float32).round().astype(np.int32)
        if len(poly) < 3:
            continue
        cv2.polylines(img, [poly], True, COLORS[s.status], 2)
        cx, cy = poly.mean(axis=0).astype(int)
        label = f"{s.index}"
        cv2.putText(img, label, (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if show_scores:   # small probability under the number, staggered so narrow slots do not overlap
            sy = cy + 24 + (10 if s.index % 2 else 0)
            cv2.putText(img, f"{s.score:.2f}", (cx - 12, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, f"{s.score:.2f}", (cx - 12, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if show_detections:
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = [int(round(v)) for v in d.box]
            color = DET_COLORS.get(d.name, (255, 0, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            if d.cls in config.VEHICLE_CLASSES and d.contact:
                cv2.circle(img, (int(d.contact[0]), int(d.contact[1])), 5, color, -1)
            tag = f"{d.name[:3]} {d.conf:.2f}" + (f" -> {assignments[i][-3:]}" if assignments and i in assignments else "")
            cv2.putText(img, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    banner = f"{status.camera_id}  free {status.free}/{status.slot_count}" + ("  (estimated)" if status.estimated else "")
    if status.registration:
        banner += "  aligned" if status.registration.get("ok") else "  NOT ALIGNED"
    if status.message:
        banner += f"  | {status.message[:60]}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def blurred_jpeg(frame: np.ndarray, dets: list[Detection], max_width: int = 1280) -> bytes:
    """Plain blurred frame (no overlays) for the admin layout editor / VLM upload."""
    img = blur_boxes(frame, dets, inplace=False)
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


STAGES = ("camera", "hooks", "bikes")


def stages(frame: np.ndarray, status: CameraStatus, dets: list[Detection], probs: dict[str, float]) -> tuple[dict, dict]:
    """The admin page's three live pictures (people blurred): what the camera sent, the hook classifier's own answer
    per hook with its score, and the final answer with the vehicles the bike finder found. Returns (jpegs, counts)."""
    from .slotcls.smoothing import vote_from_prob

    blurred = blur_boxes(frame, dets, inplace=False)
    vehicles = [d for d in dets if d.cls in config.VEHICLE_CLASSES]
    hooks = status.model_copy(deep=True)
    for s in hooks.slots:
        vote = vote_from_prob(probs.get(s.slot_id))
        s.status = "unknown" if vote is None else "occupied" if vote else "free"
    counts = {k: sum(s.status == k for s in hooks.slots) for k in ("free", "occupied", "unknown")}
    hooks = hooks.model_copy(update={**counts, "message": "model 1: hook classifier, one answer per hook"})
    bikes = status.model_copy(update={"message": f"model 2: {len(vehicles)} bike(s) found"})
    jpegs = {"camera": blurred_jpeg(frame, dets, max_width=config.PREVIEW_MAX_WIDTH),
             "hooks": render(blurred, hooks, [], show_detections=False, show_scores=True),
             "bikes": render(blurred, bikes, vehicles)}
    return jpegs, {"hooks": counts, "bikes": {"free": status.free, "occupied": status.occupied, "unknown": status.unknown,
                                              "vehicles": len(vehicles)}}
