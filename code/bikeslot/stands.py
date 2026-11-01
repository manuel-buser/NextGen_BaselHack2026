"""Rack anchor detection: one fixed stand / stirrup / wheel holder per slot (calibration, best effort).

Text prompts are weak on thin stands, so the best text-prompt exemplar (not overlapping a bike box) becomes a YOLOE
visual prompt for all buffered frames; boxes are clustered along the row. Returns anchors in working-frame pixels.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from . import config
from .schemas import Detection

STAND_PROMPTS = ["bicycle stand", "metal bicycle rack hoop", "bike wheel holder"]
_model = None
_lock = threading.Lock()
_info: dict = {"loaded": False, "model": config.STAND_MODEL, "error": None}


def info() -> dict:
    return dict(_info)


def _load():
    global _model
    with _lock:
        if _model is not None or _info["error"]:
            return _model
        try:
            from ultralytics import YOLOE

            model = YOLOE(str(config.ROOT / config.STAND_MODEL))
            model.set_classes(STAND_PROMPTS, model.get_text_pe(STAND_PROMPTS))
            _model = model
            _info.update(loaded=True)
        except Exception as exc:  # weights / text encoder download failed, no GPU, ...
            _info.update(error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return _model


def _boxes(res) -> list[tuple[list[float], float]]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    return [([float(v) for v in xyxy[i]], float(conf[i])) for i in range(len(conf))]


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _overlaps_vehicle(box, vehicles: list[Detection], thr: float = 0.3) -> bool:
    for v in vehicles:
        if _iou(box, v.box) > thr:
            return True
        # box mostly inside a vehicle box (the text prompt likes bike frames)
        ix1, iy1, ix2, iy2 = max(box[0], v.box[0]), max(box[1], v.box[1]), min(box[2], v.box[2]), min(box[3], v.box[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > 0 and inter / area > 0.6:
            return True
    return False


def detect_stands(frames: list[np.ndarray], vehicles_per_frame: list[list[Detection]]) -> Optional[dict]:
    """Find rack anchors across frames. Returns {"anchors": [...], "exemplar": box, "notes": str} or None."""
    model = _load()
    if model is None:
        return None
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    dev = config.DEVICE
    # 1. exemplar from text prompts, on the frame where it is most confident
    exemplar, ex_frame = None, None
    for idx, (f, vs) in enumerate(zip(frames, vehicles_per_frame)):
        with _lock:
            model.predictor = None  # make sure the text-prompt predictor is used
            try:
                res = model.predict(f, imgsz=config.IMGSZ, conf=0.05, iou=0.5, device=dev, verbose=False)[0]
            except Exception:
                res = model.predict(f, imgsz=config.IMGSZ, conf=0.05, iou=0.5, device="cpu", verbose=False)[0]
        for box, c in _boxes(res):
            if _overlaps_vehicle(box, vs):
                continue
            w, h = box[2] - box[0], box[3] - box[1]
            if w <= 4 or h <= 4 or w > 0.25 * f.shape[1] or h > 0.6 * f.shape[0]:
                continue
            if exemplar is None or c > exemplar[1]:
                exemplar, ex_frame = (box, c), idx
    _info["last"] = {"exemplar": None if exemplar is None else round(exemplar[1], 2), "raw": 0, "anchors": 0}
    if exemplar is None or exemplar[1] < config.STAND_EXEMPLAR_MIN_CONF:
        return None             # a weak exemplar (some random object) would only produce look-alike junk

    # 2. visual prompt on every frame (prompt image = frame of the exemplar)
    raw: list[tuple[list[float], float, int]] = []
    vp = dict(bboxes=np.array([exemplar[0]], dtype=np.float32), cls=np.array([0]))
    for idx, (f, vs) in enumerate(zip(frames, vehicles_per_frame)):
        with _lock:
            model.predictor = None
            kw = dict(visual_prompts=vp, predictor=YOLOEVPSegPredictor, imgsz=config.IMGSZ,
                      conf=config.STAND_CONF, iou=0.5, device=dev, verbose=False)
            if idx != ex_frame:
                kw["refer_image"] = frames[ex_frame]
            try:
                res = model.predict(f, **kw)[0]
            except Exception as exc:
                _info["error"] = f"visual prompt: {type(exc).__name__}: {str(exc)[:120]}"
                continue
        ew, eh = exemplar[0][2] - exemplar[0][0], exemplar[0][3] - exemplar[0][1]
        for box, c in _boxes(res):
            w, h = box[2] - box[0], box[3] - box[1]
            if not (0.4 * ew <= w <= 2.5 * ew and 0.4 * eh <= h <= 2.5 * eh):
                continue
            if _iou(box, exemplar[0]) < 0.9 and _overlaps_vehicle(box, vs, thr=0.5):
                continue
            raw.append((box, c, idx))
    with _lock:
        model.predictor = None
    _info["last"]["raw"] = len(raw)
    if len(raw) < 2:
        return None

    # 3. cluster along x (centre of the box base)
    raw.sort(key=lambda r: (r[0][0] + r[0][2]) / 2.0)
    med_w = float(np.median([r[0][2] - r[0][0] for r in raw]))
    clusters: list[list] = [[raw[0]]]
    for r in raw[1:]:
        cx = (r[0][0] + r[0][2]) / 2.0
        pcx = float(np.mean([(q[0][0] + q[0][2]) / 2.0 for q in clusters[-1]]))
        if cx - pcx > max(0.8 * med_w, 12.0):
            clusters.append([r])
        else:
            clusters[-1].append(r)
    anchors = []
    for cl in clusters:
        best = max(cl, key=lambda r: r[1])
        boxes = np.array([r[0] for r in cl])
        anchors.append({"x": float(np.mean((boxes[:, 0] + boxes[:, 2]) / 2.0)), "y": float(np.mean(boxes[:, 3])),
                        "w": float(np.mean(boxes[:, 2] - boxes[:, 0])), "h": float(np.mean(boxes[:, 3] - boxes[:, 1])),
                        "conf": float(best[1]), "hits": len({r[2] for r in cl}), "inferred": False})
    anchors = [a for a in anchors if a["conf"] >= config.STAND_CONF]
    n_detected = len(anchors)
    _info["last"]["anchors"] = n_detected
    if n_detected < 2:
        return None

    # hidden stands are not invented here: calibrate.py fits a projective lattice to the detected ones
    notes = f"stands: exemplar conf {exemplar[1]:.2f} (text prompt), {n_detected} detected by visual prompt"
    return {"anchors": anchors, "exemplar": exemplar[0], "notes": notes, "detected": n_detected}


def pitch_estimate(xs: list[float]) -> float:
    """Typical single-slot spacing: the lower part of the gap distribution (big gaps hide missing stands)."""
    gaps = np.diff(sorted(xs))
    if len(gaps) == 0:
        return 0.0
    return float(np.percentile(gaps, 35))


def _fill_gaps(anchors: list[dict]) -> list[dict]:
    """Insert inferred anchors where a gap spans k > 1 pitches (stands hidden behind bikes). The pitch varies with
    perspective, so k makes gap/k closest to the neighbouring clean gaps, not to a global median."""
    anchors = sorted(anchors, key=lambda a: a["x"])
    if len(anchors) < 3:
        return anchors
    xs = [a["x"] for a in anchors]
    gaps = np.diff(xs)
    ref = pitch_estimate(xs)
    clean = [(i, g) for i, g in enumerate(gaps) if 0.6 * ref <= g <= 1.5 * ref]
    if len(clean) < 2 or len(clean) < 0.5 * len(gaps):
        return anchors          # spacing too irregular to trust -> do not invent stands
    budget = len(anchors)       # never infer more stands than were actually seen
    out = [anchors[0]]
    for i in range(1, len(anchors)):
        gap = float(gaps[i - 1])
        left = [g for j, g in clean if j < i - 1]
        right = [g for j, g in clean if j > i - 1]
        neigh = ([left[-1]] if left else []) + ([right[0]] if right else [])
        pitch = float(np.mean(neigh)) if neigh else ref
        k = 1
        if pitch > 0:
            k = int(min(range(1, 5), key=lambda kk: abs(gap / kk - pitch)))
        k = min(k, budget + 1)
        budget -= max(0, k - 1)
        if k >= 2:
            prev = out[-1]
            for j in range(1, k):
                t = j / k
                out.append({"x": prev["x"] + (anchors[i]["x"] - prev["x"]) * t,
                            "y": prev["y"] + (anchors[i]["y"] - prev["y"]) * t,
                            "w": prev["w"], "h": prev["h"], "conf": 0.0, "hits": 0, "inferred": True})
        out.append(anchors[i])
    return out
