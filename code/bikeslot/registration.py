"""Align every picture of a camera onto its reference frame, so a slot layout drawn once survives camera bumps.

SIFT + ratio test + RANSAC homography (similarity as fallback), plausibility checks, then warpPerspective.
Features are found at REG_WORK_WIDTH (half the 640 px camera picture) after CLAHE, lamps masked, so only coarse
structure is matched. Privacy: a ``Reference`` holds keypoints and descriptors only, never pixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


@dataclass
class Reference:
    size: tuple[int, int]        # (w, h) of the frame the layout was drawn on
    points: np.ndarray           # (N, 2) float32 keypoint positions in pixels of that frame
    descriptors: np.ndarray      # (N, 128) float32 SIFT descriptors

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(f, size=np.array(self.size), points=self.points, descriptors=self.descriptors)

    @classmethod
    def load(cls, path: Path) -> "Reference":
        with np.load(path) as d:
            return cls((int(d["size"][0]), int(d["size"][1])), d["points"].astype(np.float32),
                       d["descriptors"].astype(np.float32))

    def info(self) -> dict:
        return {"size": list(self.size), "features": int(len(self.points))}


@dataclass
class Registration:
    ok: bool
    reason: str = ""
    H: Optional[np.ndarray] = None        # 3x3, picture pixels -> reference pixels
    model: str = "none"                   # homography | similarity | none
    matches: int = 0
    inliers: int = 0
    scale: Optional[float] = None         # relative to the reference, after evening out the resolutions
    rotation_deg: Optional[float] = None
    shift_px: Optional[list[float]] = None
    perspective: Optional[float] = None
    ms: float = 0.0

    def info(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "model": self.model, "matches": self.matches,
                "inliers": self.inliers, "scale": self.scale, "rotation_deg": self.rotation_deg,
                "shift_px": self.shift_px, "perspective": self.perspective, "ms": self.ms}


def features(img: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Keypoint positions (pixels of ``img``) and SIFT descriptors, found at REG_WORK_WIDTH."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = config.REG_WORK_WIDTH / gray.shape[1]
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    s = min(s, 1.0)
    small = _CLAHE.apply(small)
    burnt = cv2.dilate((small > 245).astype(np.uint8), np.ones((9, 9), np.uint8))
    sift = cv2.SIFT_create(nfeatures=config.REG_FEATURES, contrastThreshold=0.01)
    kp, des = sift.detectAndCompute(small, ((burnt == 0) * 255).astype(np.uint8))
    return np.float32([k.pt for k in kp]).reshape(-1, 2) / s, des


def make_reference(img: np.ndarray) -> Reference:
    points, des = features(img)
    if des is None or len(points) < config.REG_MIN_INLIERS:
        raise ValueError(f"only {len(points)} features in the reference picture - is the camera covered or in the dark?")
    return Reference((img.shape[1], img.shape[0]), points, des)


def _measure(reg: Registration, H: np.ndarray, picture_size: tuple[int, int], ref: Reference) -> None:
    """Scale, rotation, shift and perspective of H after evening out the two resolutions."""
    even = np.diag([picture_size[0] / ref.size[0], picture_size[1] / ref.size[1], 1.0])
    Hn = H @ even
    Hn = Hn / Hn[2, 2]
    A = Hn[:2, :2]
    reg.scale = round(float(np.sqrt(abs(np.linalg.det(A)))), 3)
    reg.rotation_deg = round(float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))), 1)
    centre = np.float32([[[ref.size[0] / 2, ref.size[1] / 2]]])
    moved = cv2.perspectiveTransform(centre, Hn)[0, 0] - centre[0, 0]
    reg.shift_px = [round(float(moved[0]), 1), round(float(moved[1]), 1)]
    reg.perspective = round(float(max(abs(Hn[2, 0]), abs(Hn[2, 1]))), 5)


def _implausible(reg: Registration, H: np.ndarray) -> str:
    if reg.inliers < config.REG_MIN_INLIERS:
        return f"only {reg.inliers} matches agree (need {config.REG_MIN_INLIERS})"
    if np.linalg.det(H[:2, :2]) <= 0:
        return "mirrored transform"
    if not config.REG_MIN_SCALE <= reg.scale <= config.REG_MAX_SCALE:
        return f"scale {reg.scale} outside {config.REG_MIN_SCALE}-{config.REG_MAX_SCALE}"
    if abs(reg.rotation_deg) > config.REG_MAX_ROTATION_DEG:
        return f"rotated {reg.rotation_deg} deg (max {config.REG_MAX_ROTATION_DEG})"
    if reg.perspective > config.REG_MAX_PERSPECTIVE:
        return f"perspective change {reg.perspective} too strong (max {config.REG_MAX_PERSPECTIVE})"
    return ""


def register(ref: Reference, img: np.ndarray) -> Registration:
    """Transform from ``img`` onto the reference frame, or the reason there is none."""
    t0 = time.time()
    reg = _register(ref, img)
    reg.ms = round((time.time() - t0) * 1000, 1)
    return reg


def _register(ref: Reference, img: np.ndarray) -> Registration:
    points, des = features(img)
    if des is None or len(points) < config.REG_MIN_INLIERS:
        return Registration(False, f"only {len(points)} features in the picture")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, ref.descriptors, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < config.REG_RATIO * m[1].distance]
    if len(good) < config.REG_MIN_INLIERS:
        return Registration(False, f"only {len(good)} features match the reference", matches=len(good))
    src = np.float32([points[m.queryIdx] for m in good])
    dst = np.float32([ref.points[m.trainIdx] for m in good])
    size = (img.shape[1], img.shape[0])
    reason = ""
    for model in ("homography", "similarity"):
        if model == "homography":
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, config.REG_RANSAC_PX)
        else:
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=config.REG_RANSAC_PX)
            H = None if M is None else np.vstack([M, [0.0, 0.0, 1.0]])
        if H is None:
            reason = reason or f"no {model} fits the matches"
            continue
        reg = Registration(False, H=H, model=model, matches=len(good), inliers=int(inl.sum()))
        _measure(reg, H, size, ref)
        problem = _implausible(reg, H)
        if not problem:
            reg.ok = True
            return reg
        reason = reason or problem
    return Registration(False, reason, matches=len(good))


def warp(img: np.ndarray, reg: Registration, ref: Reference) -> tuple[np.ndarray, np.ndarray]:
    """The picture warped onto the reference frame, and a uint8 mask (1 = the camera sees this pixel)."""
    aligned = cv2.warpPerspective(img, reg.H, ref.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    seen = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), reg.H, ref.size, flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
    return aligned, seen
