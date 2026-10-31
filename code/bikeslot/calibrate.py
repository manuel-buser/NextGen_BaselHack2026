"""Stage A: propose a slot layout for a camera from its buffered frames.

Local evidence (works offline, on any rack once it has been used):
    vehicles detected in the last frames -> ground-contact points -> 1-D clusters along the row
    -> one cluster per used slot, the row extent, and a line fit through the contact points (front edge).
    The back edge is placed BACK_OFFSET_FRAC bike heights behind it.
    The largest number of vehicles seen in one frame is a hard lower bound on the slot count.
VLM evidence (optional): parking type, enumerated slots (free/occupied), numbering direction, row corners.
    "vehicles detected in the last frame + free slots the VLM sees in it" is another lower bound on N.
Fusion: N = max of all estimates; the local row extent is widened only as far as the VLM corners support,
otherwise the missing slots are assumed to be interior and the extent is simply re-divided.
Without any vehicle the VLM corners are used.  Result: Layout(status="proposed") for the admin to confirm.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config, detector, vlm
from .geometry import normalize, slots_from_rows
from .preview import blurred_jpeg
from .schemas import CalibrationHint, Detection, Layout, Row, Slot, utcnow

BACK_OFFSET_FRAC = 0.6     # back line = front line shifted up by this share of the median bike height
FRONT_PAD_FRAC = 0.08      # front line pushed towards the camera by this share of the bike height
MIN_CONF_FOR_LAYOUT = 0.3  # detections below this are ignored for calibration


def propose_layout(cam, use_vlm: bool = True) -> Layout:
    """``cam`` is a store.CameraState (duck-typed to avoid an import cycle)."""
    frames = [f for _, f in cam.frames]
    if not frames:
        raise ValueError("no frames buffered for this camera")
    h, w = frames[-1].shape[:2]
    frame_size = (w, h)

    vehicles: list[Detection] = []
    per_frame: list[int] = []
    last_dets: list[Detection] = []
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        dets = detector.detect(f)
        last_dets = dets
        vs = [d for d in dets if d.cls in config.VEHICLE_CLASSES and d.conf >= MIN_CONF_FOR_LAYOUT]
        vehicles.extend(vs)
        per_frame.append(len(vs))

    local = local_proposal(vehicles, frame_size)
    hint: Optional[CalibrationHint] = None
    if use_vlm:
        hint = vlm.describe_scene(blurred_jpeg(frames[-1], last_dets, max_width=config.VLM_MAX_SIDE))
    painted = painted_dividers(frames, local, frame_size) if local is not None else None

    layout = fuse(cam.camera_id, frame_size, local, hint, n_frames=len(frames),
                  max_per_frame=max(per_frame) if per_frame else 0, last_frame_vehicles=per_frame[-1] if per_frame else 0,
                  painted=painted)
    layout.site_id = cam.layout.site_id
    layout.site_name = cam.layout.site_name
    return layout


# --------------------------------------------------------------------------- local evidence

def local_proposal(vehicles: list[Detection], frame_size: tuple[int, int]) -> Optional[dict]:
    """Cluster vehicle contact points along x. Returns pixel-space row geometry or None without vehicles."""
    if not vehicles:
        return None
    pts = np.array([d.contact or d.bottom_center for d in vehicles], dtype=np.float64)
    widths = np.array([d.width for d in vehicles])
    heights = np.array([d.height for d in vehicles])
    med_w, med_h = float(np.median(widths)), float(np.median(heights))

    order = np.argsort(pts[:, 0])
    xs = pts[order, 0]
    gap = config.CLUSTER_GAP_FRAC * med_w
    clusters: list[list[int]] = [[int(order[0])]]
    for prev_x, i in zip(xs[:-1], order[1:]):
        if pts[i, 0] - prev_x > gap:
            clusters.append([int(i)])
        else:
            clusters[-1].append(int(i))
    centres = np.array([pts[c].mean(axis=0) for c in clusters])
    n_local = len(clusters)
    spacing = float(np.median(np.diff(centres[:, 0]))) if n_local >= 2 else med_w * 1.1

    # front line through the contact points (least squares y = a x + b); horizontal if too few points
    if len(pts) >= 2 and np.ptp(pts[:, 0]) > 1.0:
        a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    else:
        a, b = 0.0, float(pts[:, 1].mean())
    return {"n_local": n_local, "spacing": spacing, "a": float(a), "b": float(b),
            "first": float(centres[0, 0]), "last": float(centres[-1, 0]),
            "med_w": med_w, "med_h": med_h, "centres": centres.tolist(), "vehicles": len(vehicles)}


# --------------------------------------------------------------------------- painted dividers

def painted_dividers(frames: list[np.ndarray], local: dict, frame_size: tuple[int, int]) -> Optional[dict]:
    """Look for painted bay dividers inside a generous quad around the bikes (see lines.py)."""
    from . import lines

    w, h = frame_size
    n_local = local["n_local"]
    pitch = (local["last"] - local["first"]) / (n_local - 1) if n_local >= 2 else local["spacing"]
    left = max(0.0, local["first"] - 1.5 * pitch)
    right = min(float(w - 1), local["last"] + 1.5 * pitch)
    a, b, med_h = local["a"], local["b"], local["med_h"]

    def y_at(x: float, dy: float) -> float:
        return float(min(h - 1, max(0.0, a * x + b + dy)))

    quad = [[left, y_at(left, -1.1 * med_h)], [right, y_at(right, -1.1 * med_h)],
            [right, y_at(right, 0.15 * med_h)], [left, y_at(left, 0.15 * med_h)]]
    try:
        return lines.find_dividers(frames[-5:], quad)
    except Exception as exc:  # never let the refinement break calibration
        print(f"[calibrate] divider detection failed: {exc}")
        return None


# --------------------------------------------------------------------------- fusion

def fuse(camera_id: str, frame_size: tuple[int, int], local: Optional[dict], hint: Optional[CalibrationHint],
         n_frames: int, max_per_frame: int, last_frame_vehicles: int, painted: Optional[dict] = None) -> Layout:
    w, h = frame_size
    parking_type = hint.parking_type if hint else "unknown"
    direction = hint.row_direction if hint else "left_to_right"
    notes: list[str] = [f"{n_frames} frame(s), max {max_per_frame} vehicle(s) in one frame"]

    estimates = {"max_vehicles_per_frame": max_per_frame}
    if local is not None:
        estimates["local_clusters"] = local["n_local"]
    if hint:
        estimates["vlm_count"] = hint.slot_count
        if hint.free_count is not None:
            estimates["last_frame_vehicles_plus_vlm_free"] = last_frame_vehicles + hint.free_count
    if painted and painted["spacing_cv"] <= 0.25:
        # painted dividers are an extra count estimate only (experimental: they under-count when bikes hide
        # lines, so they never override the row geometry)
        estimates["painted_dividers"] = painted["count"]
        if parking_type in ("unknown", "unmarked"):
            parking_type = "painted_bays"
    n = max(estimates.values()) if estimates else 0
    notes.append("slot count estimates " + ", ".join(f"{k}={v}" for k, v in estimates.items()) + f" -> N={n}")

    if local is None and not (hint and hint.corners and n > 0):
        raise ValueError("cannot propose a layout: no vehicles detected and no usable VLM answer - "
                         "park a bike in the row, post a few more frames, or draw the row corners manually")

    if local is not None:
        n = max(n, 1)
        n_local = local["n_local"]
        a, b = local["a"], local["b"]
        pitch = (local["last"] - local["first"]) / (n_local - 1) if n_local >= 2 else local["spacing"]
        left = local["first"] - pitch / 2.0
        right = local["last"] + pitch / 2.0
        extra = n - n_local
        if extra > 0:
            if hint and hint.corners:
                # widen only where the VLM sees more row than the bikes cover; the rest is interior
                vl = min(hint.corners[0][0], hint.corners[3][0]) * w
                vr = max(hint.corners[1][0], hint.corners[2][0]) * w
                add_left = min(extra, int(round(max(0.0, left - vl) / pitch)))
                add_right = min(extra - add_left, int(round(max(0.0, vr - right) / pitch)))
            else:
                add_left = extra // 2
                add_right = extra - add_left
            left -= add_left * pitch
            right += add_right * pitch
            interior = extra - add_left - add_right
            notes.append(f"local clusters {n_local} -> widened by {add_left} left / {add_right} right, "
                         f"{interior} missing slot(s) assumed interior")
        left, right = max(0.0, left), min(float(w - 1), right)
        front_pad = FRONT_PAD_FRAC * local["med_h"]
        back_off = BACK_OFFSET_FRAC * local["med_h"]

        def y_at(x: float, dy: float) -> float:
            return float(min(h - 1, max(0.0, a * x + b + dy)))

        front = [[left, y_at(left, front_pad)], [right, y_at(right, front_pad)]]
        back = [[left, y_at(left, -back_off)], [right, y_at(right, -back_off)]]
    else:
        c = [[x * w, y * h] for x, y in hint.corners]  # back-left, back-right, front-right, front-left
        back, front = [c[0], c[1]], [c[3], c[2]]
        notes.append(f"no vehicles detected: row corners and slot count {n} taken from the VLM")

    row = Row(row_id="r1", slot_count=int(n), front=normalize(front, frame_size), back=normalize(back, frame_size),
              direction=direction)
    occupancy_mode = "count_only" if parking_type == "dense_rack" else "per_slot"
    if hint:
        notes.append(hint.notes)
    calibration_hint = CalibrationHint(source=hint.source if hint else "local", parking_type=parking_type,
                                       slot_count=int(n), row_direction=direction,
                                       corners=hint.corners if hint else None,
                                       free_count=hint.free_count if hint else None,
                                       occupied_count=hint.occupied_count if hint else None,
                                       notes=" | ".join(notes))
    layout = Layout(camera_id=camera_id, frame_size=[w, h], parking_type=parking_type, status="proposed",
                    source="auto", occupancy_mode=occupancy_mode, rows=[row], slots=[],
                    calibration_hint=calibration_hint, updated_at=utcnow())
    layout.slots = slots_from_rows(layout)
    return layout
