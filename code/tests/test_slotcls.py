"""Slot-classifier approach: crops, smoothing, demo layout/labels, model round trip, pipeline with a stub model."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from bikeslot import config
from bikeslot.schemas import Layout
from bikeslot.slotcls import pipeline, smoothing
from bikeslot.slotcls.crops import crop_polygon, slot_crops
from bikeslot.slotcls.model import SlotClassifier, parse_size, train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from slotcls_calibrate import layout_from_corners  # noqa: E402

DEMO_DIR = config.ROOT / "assets" / "pictures" / "DemoPictures"


def test_crop_rectifies_a_tilted_quad():
    img = np.zeros((300, 400, 3), np.uint8)
    quad = np.array([[100, 50], [250, 80], [230, 240], [80, 210]], np.int32)
    cv2.fillPoly(img, [quad], (0, 200, 0))
    crop = crop_polygon(img, quad.astype(float), size=96)
    assert crop.shape == (96, 96, 3)
    inner = crop[8:-8, 8:-8]
    assert (inner[:, :, 1] > 150).mean() > 0.98        # the quad fills the patch
    tall = crop_polygon(img, quad.astype(float), size=(64, 160), margin=0.5)
    assert tall.shape == (160, 64, 3)
    assert (tall[0, :, 1] < 50).all()                  # with margin the surroundings (black) come in


def test_layout_from_corners_splits_in_order():
    layout = layout_from_corners("camX", (2000, 1500), [[100, 500], [1900, 600], [1950, 900], [50, 800]], 18)
    assert len(layout.slots) == 18 and layout.status == "confirmed" and layout.params["engine"] == "classifier"
    xs = [s.polygon[0][0] for s in layout.slots]
    assert xs == sorted(xs) and layout.slots[0].slot_id == "camX-S01" and layout.slots[-1].index == 18
    assert all(0.0 <= v <= 1.0 for s in layout.slots for pt in s.polygon for v in pt)


def test_demo_layout_and_labels_are_consistent():
    layout = Layout.model_validate_json((DEMO_DIR / "demo_picture.layout.json").read_text(encoding="utf-8"))
    labels = json.loads((DEMO_DIR / "demo_picture.labels.json").read_text(encoding="utf-8"))
    assert layout.camera_id == "demo01" and len(layout.slots) == 19
    frame = labels["frames"][0]
    assert set(frame["slots"]) == {s.slot_id for s in layout.slots}
    assert [k for k, v in frame["slots"].items() if v == "occupied"] == [f"demo01-S{i:02d}" for i in (4, 7, 11, 15, 19)]
    assert frame["slots"]["demo01-S03"] == "unknown"  # hidden behind the bike on hook 4 from the camera position
    img = cv2.imread(str(DEMO_DIR / frame["file"]))
    crops = slot_crops(layout, img, size=96, margin=0.15)
    assert len(crops) == 19 and all(c.shape == (96, 96, 3) for c in crops.values())


def test_vote_band_and_hysteresis():
    assert smoothing.vote_from_prob(0.9) == 1.0 and smoothing.vote_from_prob(0.1) == 0.0
    assert smoothing.vote_from_prob(0.55, unsure_band=0.1) is None and smoothing.vote_from_prob(None) is None
    layout = layout_from_corners("c", (100, 100), [[0, 0], [100, 0], [100, 50], [0, 50]], 1)
    layout.params.update(k_frames=5, min_frames=2)
    trackers = smoothing.make_trackers(layout)
    sid = layout.slots[0].slot_id
    smoothing.update(trackers, {sid: 0.9})
    assert trackers[sid].state == "unknown"            # one vote is not enough
    smoothing.update(trackers, {sid: 0.9})
    assert trackers[sid].state == "occupied"
    smoothing.update(trackers, {sid: 0.1})             # a single contradicting frame does not flip the state
    assert trackers[sid].state == "occupied"
    for _ in range(4):
        smoothing.update(trackers, {sid: 0.1})
    assert trackers[sid].state == "free"


def test_parse_size():
    assert parse_size(96) == (96, 96) and parse_size("64x160") == (64, 160) and parse_size([3, 4]) == (3, 4)


def test_model_learns_a_toy_problem_and_round_trips(tmp_path):
    """Frozen ImageNet backbone, bright vs dark patches: the head must separate them, save/load must agree."""
    import torch

    torch.manual_seed(0)                                 # before the model: the head's init must be deterministic
    try:
        model = SlotClassifier("mobilenet_v3_small", 96, pretrained=True)
    except Exception as exc:  # pragma: no cover - torchvision weights not cached and no network
        pytest.skip(f"ImageNet weights unavailable: {exc}")
    rng = np.random.default_rng(0)
    patches, labels = [], []
    for i in range(32):
        y = i % 2
        base = 200 if y else 40
        patches.append(np.clip(rng.normal(base, 15, (96, 96, 3)), 0, 255).astype(np.uint8))
        labels.append(y)

    class Toy(torch.utils.data.Dataset):
        def __init__(self, model):
            self.model = model

        def __len__(self):
            return len(patches)

        def items(self):
            return list(zip(patches, labels))

        def __getitem__(self, i):
            return self.model.to_tensor([patches[i]])[0], torch.tensor(float(labels[i]))

    train(model, Toy(model), epochs=10, batch_size=16, lr=5e-3, device="cpu", log=lambda *_: None, min_per_epoch=64)
    p = model.predict_proba(patches)
    assert ((p >= 0.5).astype(int) == np.array(labels)).mean() == 1.0
    model.save(tmp_path / "m.pth")                       # head only: a few KB
    assert (tmp_path / "m.pth").stat().st_size < 200_000
    again = SlotClassifier.load(tmp_path / "m.pth", device="cpu")
    assert np.allclose(again.predict_proba(patches), p, atol=1e-4)


class StubModel:
    input_size = (96, 96)

    def __init__(self, occupied_ids):
        self.occupied_ids = occupied_ids

    def predict_proba(self, crops):
        return np.array([0.9 if i in self.occupied_ids else 0.1 for i in range(len(crops))], np.float32)


def test_pipeline_status_with_stub_model():
    layout = Layout.model_validate_json((DEMO_DIR / "demo_picture.layout.json").read_text(encoding="utf-8"))
    layout.params.update(k_frames=3, min_frames=1)
    img = cv2.imread(str(DEMO_DIR / "demo_picture.jpeg"))
    trackers = smoothing.make_trackers(layout)
    res = pipeline.analyze_frame(layout, img, "demo01", trackers, model=StubModel({2, 5}), blur_persons=False)
    st = res.status
    assert st.slot_count == 19 and st.occupied == 2 and st.free == 17 and st.unknown == 0
    assert st.frame_size == [1280, 960] and len(st.slots[0].polygon) == 4
    assert [s.slot_id for s in st.slots if s.status == "occupied"] == ["demo01-S03", "demo01-S06"]
    assert res.probs["demo01-S03"] == pytest.approx(0.9) and st.slots[2].score == pytest.approx(0.9)
    assert st.processing_ms > 0 and st.detections["person"] == 0


def test_pipeline_without_layout_reports_message():
    img = np.zeros((480, 640, 3), np.uint8)
    res = pipeline.analyze_frame(Layout(camera_id="new"), img, "new", model=StubModel(set()), blur_persons=False)
    assert res.status.slot_count == 0 and "calibrate" in res.status.message
