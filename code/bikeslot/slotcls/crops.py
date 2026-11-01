"""Slot polygons -> fixed-size crops.

A 4-point polygon (normalised, back-left first) is perspective-rectified to ``size`` x ``size``; more vertices fall
back to the bounding rectangle. ``margin`` grows the polygon around its centre to keep a little context.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..geometry import denormalize
from ..schemas import Layout, Point


def expand_polygon(poly: Sequence[Point], margin: float) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32)
    if margin == 0.0:
        return pts
    centre = pts.mean(axis=0)
    return (centre + (pts - centre) * (1.0 + margin)).astype(np.float32)


def crop_polygon(img: np.ndarray, poly_px: Sequence[Point], size: int | tuple[int, int] = 96,
                 margin: float = 0.0) -> np.ndarray:
    """Return the slot patch (BGR, ``size``) for a polygon given in pixel coordinates of ``img``."""
    w, h = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    pts = expand_polygon(poly_px, margin)
    if len(pts) == 4:
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], max(x2, x1 + 1)), min(img.shape[0], max(y2, y1 + 1))
    return cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_AREA)


def slot_crops(layout: Layout, img: np.ndarray, size: int | tuple[int, int] = 96,
               margin: float = 0.0) -> dict[str, np.ndarray]:
    """{slot_id: patch} for every slot of the layout, cropped from ``img`` (any resolution)."""
    fs = (img.shape[1], img.shape[0])
    return {s.slot_id: crop_polygon(img, denormalize(s.polygon, fs), size, margin) for s in layout.slots}


def contact_sheet(crops: dict[str, np.ndarray], labels: dict[str, str] | None = None,
                  probs: dict[str, float] | None = None, cols: int = 9, scale: int = 2) -> np.ndarray:
    """Tile the crops (with slot id / label / probability) for a quick visual check."""
    if not crops:
        return np.zeros((10, 10, 3), np.uint8)
    tiles = []
    for sid, c in crops.items():
        t = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        txt = sid.split("-")[-1]
        if labels and sid in labels:
            txt += " " + labels[sid][:3]
        if probs and sid in probs:
            txt += f" {probs[sid]:.2f}"
        color = (255, 255, 255)
        if labels and labels.get(sid) == "occupied":
            color = (80, 80, 255)
        elif labels and labels.get(sid) == "free":
            color = (80, 220, 80)
        cv2.putText(t, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        tiles.append(t)
    th, tw = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
