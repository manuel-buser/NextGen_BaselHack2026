"""Painted-bay refinement for calibration: find the painted divider lines inside a rough row quad.

Two passes (rectify, detect thin bright near-vertical segments, rectify again on the outermost dividers). Returns
None for racks without ground markings; geometry is in image pixels.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

RW, RH = 1600, 400          # rectified strip size
MIN_DIVIDERS = 3            # need at least 2 bays
MAX_SPACING_CV = 0.40       # final spacing must be roughly regular (AI/real scenes are not perfect lattices)
MIN_STRENGTH = 0.30         # candidates weaker than this share of the median strength are dropped
_TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def _paint_mask(bgr: np.ndarray) -> np.ndarray:
    """Bright, unsaturated pixels (paint); the lightness threshold adapts to the strip, and a top-hat
    rescue adds thin bright lines that lie in shadow (e.g. the end bay next to a wall)."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    L, S = hls[:, :, 1], hls[:, :, 2]
    l_thr = max(140.0, float(np.percentile(L, 90)))
    bright = (L >= l_thr) & (S < 80)
    top = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, _TOPHAT)
    thin = (top >= max(40.0, float(np.percentile(top, 98)))) & (S < 80) & (L >= 90)
    mask = (bright | thin).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _rectify(frames: Sequence[np.ndarray], quad_px: Sequence[Sequence[float]]):
    """Return (H, H_inv, paint mask of the strip accumulated over frames) or None for a degenerate quad."""
    src = np.array(quad_px, dtype=np.float32)
    if abs(cv2.contourArea(src)) < 1e-3:
        return None
    dst = np.array([[0, 0], [RW, 0], [RW, RH], [0, RH]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    mask = None
    for f in frames:
        m = _paint_mask(cv2.warpPerspective(f, H, (RW, RH)))
        mask = m if mask is None else np.maximum(mask, m)
    return H, np.linalg.inv(H), mask


def _to_image(H_inv: np.ndarray, pts: Sequence[Sequence[float]]) -> list[list[float]]:
    arr = np.array([pts], dtype=np.float32)
    return [[float(x), float(y)] for x, y in cv2.perspectiveTransform(arr, H_inv)[0]]


def _candidates(mask: np.ndarray, max_tilt_deg: float, cluster_gap: float) -> list[dict]:
    """Divider candidates from Hough segments: {x_front, x_back, y_top, strength}, sorted by x_front."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=35, minLineLength=int(RH * 0.2), maxLineGap=int(RH * 0.1))
    if segs is None or len(segs) == 0:
        return []
    cands = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):   # OpenCV 4 gives Nx1x4, OpenCV 5 gives Nx4
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dy) < 1e-6 or np.degrees(np.arctan2(abs(dx), abs(dy))) > max_tilt_deg:
            continue
        slope = dx / dy
        cands.append((x1 + (RH - y1) * slope, x1 + (0 - y1) * slope, float(min(y1, y2)), float(np.hypot(dx, dy))))
    cands.sort(key=lambda c: c[0])
    clusters: list[list] = []
    for c in cands:
        if clusters and c[0] - clusters[-1][-1][0] <= cluster_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    out = []
    for cl in clusters:
        wts = np.array([c[3] for c in cl])
        out.append({"x_front": float(np.average([c[0] for c in cl], weights=wts)),
                    "x_back": float(np.average([c[1] for c in cl], weights=wts)),
                    "y_top": float(np.percentile([c[2] for c in cl], 20)),
                    "strength": float(wts.sum())})
    return out


def _strong(cands: list[dict]) -> list[dict]:
    if not cands:
        return []
    med = float(np.median([c["strength"] for c in cands]))
    return [c for c in cands if c["strength"] >= MIN_STRENGTH * med]


def find_dividers(frames: Sequence[np.ndarray], quad_px: Sequence[Sequence[float]]) -> Optional[dict]:
    """Detect painted dividers inside ``quad_px`` (back-left, back-right, front-right, front-left, pixels).
    Returns {"front", "back" (one point per divider, left->right), "count", "spacing_cv", "spacing_px"} or None."""
    r1 = _rectify(frames, quad_px)
    if r1 is None:
        return None
    H1, H1_inv, mask1 = r1
    c1 = _strong(_candidates(mask1, max_tilt_deg=35.0, cluster_gap=0.035 * RW))
    if len(c1) < MIN_DIVIDERS:
        return None
    pitch1 = (c1[-1]["x_front"] - c1[0]["x_front"]) / (len(c1) - 1)

    # pass 2: quad spanned by the outermost candidates plus a margin -> dividers become vertical
    m = 0.45 * pitch1
    quad2 = _to_image(H1_inv, [[c1[0]["x_back"] - m, 0.0], [c1[-1]["x_back"] + m, 0.0],
                               [c1[-1]["x_front"] + m, float(RH)], [c1[0]["x_front"] - m, float(RH)]])
    r2 = _rectify(frames, quad2)
    if r2 is None:
        return None
    H2, H2_inv, mask2 = r2
    expected_pitch = RW / (len(c1) - 1 + 0.9)
    c2 = _strong(_candidates(mask2, max_tilt_deg=12.0, cluster_gap=0.3 * expected_pitch))
    if len(c2) < MIN_DIVIDERS:
        return None
    xs = [c["x_front"] for c in c2]
    spacing = np.diff(xs)
    cv_ = float(spacing.std() / spacing.mean()) if spacing.mean() > 0 else 9.0
    if cv_ > MAX_SPACING_CV and len(c2) > MIN_DIVIDERS:
        # one stray candidate (reflection, wheel rim) can ruin the regularity - drop the weakest and retry once
        weakest = int(np.argmin([c["strength"] for c in c2]))
        c2 = c2[:weakest] + c2[weakest + 1:]
        xs = [c["x_front"] for c in c2]
        spacing = np.diff(xs)
        cv_ = float(spacing.std() / spacing.mean()) if spacing.mean() > 0 else 9.0
    if cv_ > MAX_SPACING_CV:
        return None

    # the paint ends at the back of the bays: median top end of the (vertical) divider segments
    y_back = float(np.clip(np.median([c["y_top"] for c in c2]), 0.0, RH * 0.7))
    front = _to_image(H2_inv, [[x, float(RH)] for x in xs])
    back = _to_image(H2_inv, [[x, y_back] for x in xs])
    return {"front": front, "back": back, "count": len(xs) - 1, "spacing_cv": round(cv_, 3),
            "spacing_px": [round(float(s), 1) for s in spacing], "y_back_frac": round(y_back / RH, 2)}
