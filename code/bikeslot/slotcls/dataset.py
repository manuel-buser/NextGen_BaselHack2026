"""Labelled crops for training / evaluation.

Label files are small JSON documents next to the images (one entry per frame, one label per slot - what the
labelling tool writes, what ``golden_labels.json`` looks like):

    {"camera_id": "demo01", "layout": "demo_picture.layout.json",
     "frames": [{"file": "demo_picture.jpeg", "slots": {"demo01-S01": "free", "demo01-S03": "occupied", ...}}]}

Relative paths are resolved against the label file's directory.  Slots missing from a frame's ``slots`` or
labelled ``unknown`` are skipped.  Crops are cut from the full-resolution frames at load time (nothing is
duplicated on disk): every sample keeps a *context patch* (the polygon grown by ``context`` and warped to
``ctx_scale`` times the input size) so that training-time augmentation can shift / zoom / rotate the crop
inside real surroundings instead of padding with black.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import config
from ..geometry import denormalize, imread
from ..schemas import Layout
from .crops import crop_polygon
from .model import IMAGENET_MEAN, IMAGENET_STD, parse_size

LABEL_VALUES = {"free": 0, "occupied": 1}


@dataclass
class Sample:
    frame: Path
    slot_id: str
    polygon: list            # normalised 0..1 (4 points)
    label: int               # 0 free, 1 occupied
    camera_id: str = ""
    patch: Optional[np.ndarray] = None   # context patch (BGR), filled by ``materialize``


@dataclass
class LabelFile:
    path: Path
    camera_id: str
    layout: Layout
    frames: list[dict] = field(default_factory=list)
    notes: str = ""
    data_layout: str = ""        # layout reference as written in the file (relative to the file)

    @property
    def layout_path(self) -> Path:
        return self.path.parent / self.data_layout

    @classmethod
    def load(cls, path: Path | str) -> "LabelFile":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        layout_ref = data.get("layout", "")
        layout_path = (path.parent / layout_ref) if layout_ref else None
        if layout_path is None or not layout_path.exists():
            raise FileNotFoundError(f"{path}: layout file {layout_ref!r} not found next to the label file")
        layout = Layout.model_validate_json(layout_path.read_text(encoding="utf-8"))
        lf = cls(path=path, camera_id=data.get("camera_id", layout.camera_id), layout=layout,
                 frames=list(data.get("frames", [])), notes=data.get("notes", ""))
        lf.data_layout = layout_ref
        return lf

    def save(self) -> None:
        data = {"camera_id": self.camera_id, "layout": self.data_layout, "notes": self.notes, "frames": self.frames}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def samples(self) -> list[Sample]:
        polys = {s.slot_id: s.polygon for s in self.layout.slots}
        out: list[Sample] = []
        for fr in self.frames:
            frame_path = self.path.parent / fr["file"]
            for slot_id, label in fr.get("slots", {}).items():
                if label not in LABEL_VALUES or slot_id not in polys:
                    continue
                out.append(Sample(frame=frame_path, slot_id=slot_id, polygon=polys[slot_id],
                                  label=LABEL_VALUES[label], camera_id=self.camera_id))
        return out


def load_samples(label_files: Iterable[Path | str]) -> list[Sample]:
    samples: list[Sample] = []
    for p in label_files:
        samples.extend(LabelFile.load(p).samples())
    return samples


def materialize(samples: Sequence[Sample], input_size=config.SLOTCLS_INPUT, margin: float = config.SLOTCLS_MARGIN,
                context: float = 0.35, ctx_scale: float = 1.5, log=print) -> list[Sample]:
    """Read every frame once and cut the context patch of each sample (in place). Returns the usable samples."""
    w, h = parse_size(input_size)
    ctx_size = (int(round(w * ctx_scale)), int(round(h * ctx_scale)))
    by_frame: dict[Path, list[Sample]] = {}
    for s in samples:
        by_frame.setdefault(s.frame, []).append(s)
    usable: list[Sample] = []
    for frame_path, group in by_frame.items():
        img = imread(frame_path)
        if img is None:
            log(f"[slotcls] cannot read {frame_path}, skipping {len(group)} samples")
            continue
        fs = (img.shape[1], img.shape[0])
        for s in group:
            s.patch = crop_polygon(img, denormalize(s.polygon, fs), ctx_size, margin=margin + context)
            usable.append(s)
    return usable


def _centre_crop(patch: np.ndarray, w: int, h: int, context: float, margin: float) -> np.ndarray:
    """The un-augmented crop: the central part of the context patch that corresponds to margin-only growth."""
    ph, pw = patch.shape[:2]
    frac = (1.0 + margin) / (1.0 + margin + context)
    cw, ch = pw * frac, ph * frac
    x0, y0 = (pw - cw) / 2.0, (ph - ch) / 2.0
    src = np.array([[x0, y0], [x0 + cw, y0], [x0 + cw, y0 + ch], [x0, y0 + ch]], dtype=np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(patch, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_AREA)


def augment(patch: np.ndarray, w: int, h: int, context: float, margin: float, rng: random.Random) -> np.ndarray:
    """Random shift / zoom / rotation inside the context patch + photometric jitter + flip."""
    ph, pw = patch.shape[:2]
    frac = (1.0 + margin) / (1.0 + margin + context)
    zoom = frac * rng.uniform(0.85, 1.15)
    cw, ch = pw * zoom, ph * zoom
    max_dx, max_dy = (pw - cw) / 2.0, (ph - ch) / 2.0
    cx = pw / 2.0 + rng.uniform(-max_dx, max_dx) * 0.8
    cy = ph / 2.0 + rng.uniform(-max_dy, max_dy) * 0.8
    angle = rng.uniform(-6.0, 6.0)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rect = np.array([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2], [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]], dtype=np.float32)
    src = cv2.transform(rect[None], M)[0].astype(np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    out = cv2.warpPerspective(patch, cv2.getPerspectiveTransform(src, dst), (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    # photometric: brightness / contrast / colour cast / blur / noise
    out = out.astype(np.float32)
    out = (out - 128.0) * rng.uniform(0.75, 1.25) + 128.0 + rng.uniform(-30, 30)
    out += np.array([rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)], dtype=np.float32)
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        out += np.random.default_rng(rng.getrandbits(32)).normal(0, 6, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class CropDataset(Dataset):
    """(tensor 3xHxW, label) pairs from materialized samples; augmentation on when ``train``."""

    def __init__(self, samples: Sequence[Sample], input_size=config.SLOTCLS_INPUT, margin: float = config.SLOTCLS_MARGIN,
                 context: float = 0.35, train: bool = False, seed: int = 0):
        self.samples = [s for s in samples if s.patch is not None]
        self.w, self.h = parse_size(input_size)
        self.margin, self.context, self.train = margin, context, train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def items(self):
        return [(s, s.label) for s in self.samples]

    def crop(self, i: int) -> np.ndarray:
        s = self.samples[i]
        if self.train:
            return augment(s.patch, self.w, self.h, self.context, self.margin, self.rng)
        return _centre_crop(s.patch, self.w, self.h, self.context, self.margin)

    def __getitem__(self, i: int):
        c = self.crop(i)[:, :, ::-1].astype(np.float32) / 255.0
        c = (c - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(c.transpose(2, 0, 1))), torch.tensor(float(self.samples[i].label))
