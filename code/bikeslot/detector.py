"""YOLO11 wrapper: one lazily loaded model, thread-safe predict, returns plain Detection objects.

Works with a segmentation model (yolo11x-seg.pt, preferred: masks give the true ground-contact point of
each wheel) and with a detection-only model (yolo11x.pt: contact point = bottom-centre of the box).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from . import config
from .geometry import footprint_rect
from .schemas import Detection

_model = None
_precision: dict = {}
_lock = threading.Lock()
_info: dict = {"loaded": False, "model": config.MODEL_NAME, "device": config.DEVICE,
               "segmentation": False, "warmup_ms": None}


def info() -> dict:
    return dict(_info)


def load() -> None:
    """Load and warm up the model once (safe to call repeatedly)."""
    global _model
    with _lock:
        if _model is not None:
            return
        from ultralytics import YOLO  # slow import, keep it local

        t = time.time()
        model = YOLO(str(config.MODEL_PATH))
        dummy = np.zeros((int(config.IMGSZ * 0.75), config.IMGSZ, 3), dtype=np.uint8)
        device, precision = config.DEVICE, {}
        if config.HALF and str(device) != "cpu":
            # ultralytics >= 8.4 prefers quantize="fp16"; older versions only know half=True
            for candidate in ({"quantize": "fp16"}, {"half": True}):
                try:
                    model.predict(dummy, imgsz=config.IMGSZ, device=device, verbose=False, **candidate)
                    precision = candidate
                    break
                except Exception:
                    continue
        try:
            model.predict(dummy, imgsz=config.IMGSZ, device=device, verbose=False, **precision)
        except Exception:  # no CUDA -> CPU fallback
            device, precision = "cpu", {}
            model.predict(dummy, imgsz=config.IMGSZ, device=device, verbose=False)
        _model = model
        _precision.clear()
        _precision.update(precision)
        _info.update(loaded=True, device=device, precision=precision or {"fp32": True},
                     segmentation=(model.task == "segment"), warmup_ms=round((time.time() - t) * 1000))


def detect(img: np.ndarray) -> list[Detection]:
    """Run the detector on a BGR frame (already resized) and return person/bicycle/motorcycle detections."""
    load()
    with _lock:
        res = _model.predict(img, imgsz=config.IMGSZ, conf=config.CONF, iou=config.IOU,
                             classes=list(config.CLASS_NAMES), agnostic_nms=True,
                             device=_info["device"], verbose=False, **_precision)[0]
    dets: list[Detection] = []
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    masks_xy = res.masks.xy if res.masks is not None else None
    h_img, w_img = img.shape[:2]
    for i in range(len(boxes)):
        box = [float(v) for v in xyxy[i]]
        det = Detection(cls=int(clss[i]), name=config.CLASS_NAMES.get(int(clss[i]), str(clss[i])),
                        conf=float(confs[i]), box=box)
        mask_poly: Optional[np.ndarray] = None
        if masks_xy is not None and len(masks_xy[i]) >= 3:
            mask_poly = masks_xy[i].astype(np.float32)
            det.mask = [[float(x), float(y)] for x, y in mask_poly]
        det.contact = _contact_point(det, mask_poly, (h_img, w_img))
        dets.append(det)
    return dets


def _contact_point(det: Detection, mask_poly: Optional[np.ndarray], shape: tuple[int, int]) -> list[float]:
    """Ground contact point: mean of the lowest CONTACT_FRAC share of mask pixels, else bottom-centre."""
    if mask_poly is None:
        return det.bottom_center
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2 + 1), min(shape[0], y2 + 1)
    if x2 <= x1 or y2 <= y1:
        return det.bottom_center
    local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(local, [(mask_poly - [x1, y1]).round().astype(np.int32)], 1)
    ys, xs = np.nonzero(local)
    if len(ys) == 0:
        return det.bottom_center
    ymax = ys.max()
    band = max(3, int(config.CONTACT_FRAC * (y2 - y1)))
    sel = ys >= ymax - band
    return [float(xs[sel].mean() + x1), float(ys[sel].mean() + y1)]


def mask_pixels(det: Detection, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) of the detection's mask pixels, or of its footprint rectangle when there is no mask."""
    if det.mask:
        full = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(full, [np.array(det.mask, dtype=np.float32).round().astype(np.int32)], 1)
        return np.nonzero(full)
    fx1, fy1, fx2, fy2 = footprint_rect(det.box, config.FOOTPRINT_FRAC)
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(shape[1], fx2), min(shape[0], fy2)
    if fx2 <= fx1 or fy2 <= fy1:
        return np.array([], dtype=int), np.array([], dtype=int)
    yy, xx = np.mgrid[fy1:fy2, fx1:fx2]
    return yy.ravel(), xx.ravel()


def blur_boxes(img: np.ndarray, dets: list[Detection], classes: Optional[set[int]] = None,
               inplace: bool = True) -> np.ndarray:
    """Blur every detection of the given classes (persons, motorcycles) - privacy, never skipped."""
    classes = config.BLUR_CLASSES if classes is None else classes
    out = img if inplace else img.copy()
    for d in dets:
        if d.cls not in classes:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)
        if x2 > x1 and y2 > y1:
            roi = out[y1:y2, x1:x2]
            k = max(15, (min(roi.shape[:2]) // 4) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out
