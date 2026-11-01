"""Vehicle detectors behind one interface, so the model that finds parked vehicles can be swapped by configuration.

    from bikeslot import vehicles
    dets = vehicles.detect(img)               # the one chosen by BIKESLOT_VEHICLE_DETECTOR (yolo | dino)
    dets = vehicles.get("dino").detect(img)   # a specific one

Every detector returns ``schemas.Detection`` objects in pixels of the given BGR picture, with ``contact`` set to the
point where the vehicle meets the ground (lowest mask pixels, or the bottom centre of the box).  To add a model,
subclass ``VehicleDetector`` and decorate it with ``@register("name")``.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from . import config
from .schemas import Detection

_registry: dict[str, Callable[[], "VehicleDetector"]] = {}
_instances: dict[str, "VehicleDetector"] = {}
_lock = threading.Lock()


def register(name: str):
    def wrap(cls):
        _registry[name] = cls
        cls.name = name
        return cls
    return wrap


def available() -> list[str]:
    return sorted(_registry)


def get(name: Optional[str] = None) -> "VehicleDetector":
    """The detector called ``name`` (default: BIKESLOT_VEHICLE_DETECTOR), created once and reused."""
    name = name or config.VEHICLE_DETECTOR
    if name not in _registry:
        raise ValueError(f"unknown vehicle detector {name!r}; available: {', '.join(available())}")
    with _lock:
        if name not in _instances:
            _instances[name] = _registry[name]()
        return _instances[name]


def detect(img: np.ndarray) -> list[Detection]:
    return get().detect(img)


def info() -> dict:
    return {"selected": config.VEHICLE_DETECTOR, "available": available(),
            "loaded": {n: d.info() for n, d in _instances.items()}}


class VehicleDetector:
    name = "base"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info: dict = {"loaded": False, "load_ms": None}

    def load(self) -> None:
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, **self._info}


def _bottom_centre(box: list[float]) -> list[float]:
    return [(box[0] + box[2]) / 2.0, box[3]]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contains(big: Detection, small: Detection, inside: float) -> bool:
    b, s = big.box, small.box
    ov = _area([max(b[0], s[0]), max(b[1], s[1]), min(b[2], s[2]), min(b[3], s[3])])
    return small is not big and _area(s) < _area(b) and ov >= inside * _area(s)


def _side_by_side(a: Detection, b: Detection) -> bool:
    """Two different vehicles next to each other: they share less than half the narrower one's width."""
    shared = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
    return shared < 0.5 * min(a.width, b.width)


def drop_containers(dets: list[Detection], inside: float = 0.95) -> list[Detection]:
    """Boxes (almost) fully inside a bigger box. Two or more of them side by side: the bigger box is drawn around
    several vehicles - drop it, they count one by one. Otherwise all of them are one vehicle (the same one twice, or
    a wheel / basket of it): keep the outer box, which covers the whole vehicle down to its wheels."""
    dets = list(dets)
    changed = True
    while changed:
        changed = False
        for big in sorted(dets, key=lambda d: _area(d.box), reverse=True):
            inner = [s for s in dets if _contains(big, s, inside)]
            if not inner:
                continue
            tops = [s for s in inner if not any(_contains(o, s, inside) for o in inner)]
            if any(_side_by_side(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]):
                dets.remove(big)
            else:
                dets = [d for d in dets if d not in inner]
            changed = True
            break
    return dets


def rack_area(layout, frame_size: tuple[int, int], widen_hooks: float = 1.0, depth: float = 2.2) -> np.ndarray:
    """Where a parked vehicle's wheels stand: the slot row, widened by ``widen_hooks`` hook pitches at both ends and
    reaching ``depth`` times the slot depth from the rail towards the camera. Pixel polygon (4 x 2, float32)."""
    from .geometry import denormalize

    row = layout.rows[0]
    (bl, br), (fl, fr) = (np.float32(denormalize(row.back, frame_size)), np.float32(denormalize(row.front, frame_size)))
    along = (br - bl) / max(1, row.slot_count) * widen_hooks
    bl, br, fl, fr = bl - along, br + along, fl - along, fr + along
    return np.float32([bl, br, br + depth * (fr - br), bl + depth * (fl - bl)])


def in_area(det: Detection, area: np.ndarray) -> bool:
    return cv2.pointPolygonTest(area, (float(det.contact[0]), float(det.contact[1])), False) >= 0


@register("yolo")
class YoloVehicles(VehicleDetector):
    """The service's YOLO11 model (detector.py), vehicle classes only (bicycle, motorcycle)."""

    def load(self) -> None:
        from . import detector

        detector.load()
        self._info.update(loaded=True, model=config.MODEL_NAME, device=detector.info().get("device"))

    def detect(self, img: np.ndarray) -> list[Detection]:
        from . import detector

        self.load()
        return [d for d in detector.detect(img) if d.cls in config.VEHICLE_CLASSES]


@register("dino")
class GroundingDino(VehicleDetector):
    """Grounding DINO (open-vocabulary, Hugging Face transformers): finds whatever VEHICLE_PROMPT names.

    One box per vehicle: phrases of the prompt that hit the same object are merged by class-agnostic NMS."""

    def __init__(self) -> None:
        super().__init__()
        self._model = self._processor = None
        self._device = "cpu"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers.utils import import_utils

            # the venv also sees Anaconda's scikit-learn (built for NumPy 1); DINO never needs it, so keep it out
            import_utils._sklearn_available = False
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            t0 = time.time()
            self._device = "cuda" if torch.cuda.is_available() and str(config.DEVICE) != "cpu" else "cpu"
            self._processor = AutoProcessor.from_pretrained(config.DINO_MODEL)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(config.DINO_MODEL).to(self._device).eval()
            self._info.update(loaded=True, model=config.DINO_MODEL, device=self._device, prompt=config.VEHICLE_PROMPT,
                              load_ms=round((time.time() - t0) * 1000))

    def detect(self, img: np.ndarray) -> list[Detection]:
        import torch

        self.load()
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock, torch.no_grad():
            inputs = self._processor(images=rgb, text=config.VEHICLE_PROMPT, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            res = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=config.DINO_BOX_THRESHOLD,
                text_threshold=config.DINO_TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
        boxes = res["boxes"].cpu().numpy().tolist()
        scores = res["scores"].cpu().numpy().tolist()
        labels = res.get("text_labels", res.get("labels"))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], scores,
                                config.DINO_BOX_THRESHOLD, config.VEHICLE_NMS_IOU)
        dets = []
        for i in np.array(keep).flatten():
            label = str(labels[i]).strip() or "vehicle"
            cls = config.CLS_MOTORCYCLE if "motor" in label else config.CLS_BICYCLE
            box = [max(0.0, boxes[i][0]), max(0.0, boxes[i][1]), min(float(w), boxes[i][2]), min(float(h), boxes[i][3])]
            dets.append(Detection(cls=cls, name=label, conf=float(scores[i]), box=box, contact=_bottom_centre(box)))
        return drop_containers(dets)
