"""Geometry helpers: row interpolation, (de)normalisation, slot masks, point tests, IoU."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .schemas import Layout, Point, Row, Slot

# --------------------------------------------------------------------------- basic conversions


def normalize(points: Iterable[Sequence[float]], frame_size: Sequence[int]) -> list[Point]:
    w, h = float(frame_size[0]), float(frame_size[1])
    return [[float(x) / w, float(y) / h] for x, y in points]


def denormalize(points: Iterable[Sequence[float]], frame_size: Sequence[int]) -> list[Point]:
    w, h = float(frame_size[0]), float(frame_size[1])
    return [[float(x) * w, float(y) * h] for x, y in points]


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> Point:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


# --------------------------------------------------------------------------- row model


def _row_homography(row: Row) -> Optional[np.ndarray]:
    """Homography unit square -> row quadrilateral (back-left, back-right, front-right, front-left)."""
    q0, q1 = row.back
    p0, p1 = row.front
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.array([q0, q1, p1, p0], dtype=np.float32)
    if abs(cv2.contourArea(dst)) < 1e-9:
        return None
    try:
        return cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None


def interpolate_row(row: Row, camera_id: str, start_index: int = 1) -> list[Slot]:
    """Split a row into ``row.slot_count`` quadrilaterals of equal width *on the ground*.

    The row quadrilateral is mapped to a unit square (homography), split into N equal columns there and
    mapped back, so perspective foreshortening is respected (bays further away look narrower).  Falls back
    to linear interpolation along the front/back lines for degenerate quads.
    Slot polygons are returned in vertex order back-left, back-right, front-right, front-left and are
    numbered in ``row.direction``; both lines must be given left->right as seen in the image.
    """
    n = row.slot_count
    p0, p1 = row.front
    q0, q1 = row.back
    H = _row_homography(row)
    slots: list[Slot] = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        if H is not None:
            unit = np.array([[[t0, 0.0], [t1, 0.0], [t1, 1.0], [t0, 1.0]]], dtype=np.float32)
            poly = [[float(x), float(y)] for x, y in cv2.perspectiveTransform(unit, H)[0]]
        else:
            poly = [_lerp(q0, q1, t0), _lerp(q0, q1, t1), _lerp(p0, p1, t1), _lerp(p0, p1, t0)]
        slots.append(Slot(slot_id="tmp", index=1, row_id=row.row_id, polygon=poly))
    if row.direction == "right_to_left":
        slots.reverse()
    for k, s in enumerate(slots):
        s.index = start_index + k
        s.slot_id = f"{camera_id}-S{s.index:02d}"
    return slots


def slots_from_rows(layout: Layout) -> list[Slot]:
    slots: list[Slot] = []
    for row in layout.rows:
        slots.extend(interpolate_row(row, layout.camera_id, start_index=len(slots) + 1))
    return slots


def row_from_polygons(polys: Sequence[Sequence[Point]], row_id: str = "r1") -> Row:
    """Fit a Row to an ordered list of slot polygons (back-left, back-right, front-right, front-left)."""
    first, last = polys[0], polys[-1]
    return Row(row_id=row_id, slot_count=len(polys),
               front=[list(first[3]), list(last[2])], back=[list(first[0]), list(last[1])])


# --------------------------------------------------------------------------- rasterisation


def pixel_polygon(poly: Sequence[Point], frame_size: Sequence[int]) -> np.ndarray:
    return np.array(denormalize(poly, frame_size), dtype=np.float32).round().astype(np.int32)


def slot_masks(layout: Layout, frame_size: Sequence[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {slot_id: (uint8 mask HxW, int32 polygon Nx2)} in pixel coordinates of ``frame_size``."""
    w, h = int(frame_size[0]), int(frame_size[1])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in layout.slots:
        poly = pixel_polygon(s.polygon, frame_size)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        out[s.slot_id] = (mask, poly)
    return out


def row_hull(layout: Layout, frame_size: Sequence[int]) -> Optional[np.ndarray]:
    """Convex hull (int32 polygon) around all slots - used by count_only mode."""
    if not layout.slots:
        return None
    pts = np.concatenate([pixel_polygon(s.polygon, frame_size) for s in layout.slots])
    return cv2.convexHull(pts).reshape(-1, 2)


def point_in_polygon(poly: np.ndarray, pt: Sequence[float]) -> bool:
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def polygon_area(poly: Sequence[Point]) -> float:
    arr = np.asarray(poly, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_iou(a: Sequence[Point], b: Sequence[Point]) -> float:
    """IoU of two polygons (shapely if available, else rasterised at 1000x1000 in normalised space)."""
    try:
        from shapely.geometry import Polygon

        pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
        union = pa.union(pb).area
        return float(pa.intersection(pb).area / union) if union > 0 else 0.0
    except Exception:  # pragma: no cover - fallback without shapely
        size = (1000, 1000)
        ma = np.zeros(size, np.uint8)
        cv2.fillPoly(ma, [pixel_polygon(a, size)], 1)
        mb = np.zeros(size, np.uint8)
        cv2.fillPoly(mb, [pixel_polygon(b, size)], 1)
        union = np.logical_or(ma, mb).sum()
        return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


# --------------------------------------------------------------------------- image helpers


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize so the longer side equals ``max_side`` (never upscales). Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, scale


def laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def footprint_rect(box: Sequence[float], frac: float, width_frac: float = 0.6) -> tuple[int, int, int, int]:
    """Bottom ``frac`` of the box height x central ``width_frac`` of its width -> (x1, y1, x2, y2) ints."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = (1.0 - width_frac) / 2.0 * w
    return int(round(x1 + pad)), int(round(y2 - frac * h)), int(round(x2 - pad)), int(round(y2))


def principal_axis_order(points: np.ndarray) -> np.ndarray:
    """Indices that sort 2-D points along their principal axis, oriented so that x increases."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.arange(len(pts))
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0:
        axis = -axis
    return np.argsort(centred @ axis)
