"""End-to-end check on the golden data set (needs the model weights + a GPU/CPU; ~10 s)."""
import cv2
import pytest

from bikeslot import config
from bikeslot.occupancy import SlotMaskCache, SlotTracker, analyze_frame, make_trackers
from bikeslot.store import golden_truth, layout_from_golden

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model weights not downloaded")


def test_golden_set_per_frame_accuracy():
    layout = layout_from_golden()
    layout.params = {"k_frames": 1, "min_frames": 1}
    total = correct = 0
    for item in golden_truth():
        img = cv2.imread(str(config.GOLDEN_DIR / item["file"]))
        res = analyze_frame(layout, make_trackers(layout), SlotMaskCache(), img, layout.camera_id)
        pred = {s.slot_id: s.status for s in res.status.slots}
        for sid, want in item["slots"].items():
            total += 1
            correct += pred.get(sid) == want
        assert res.status.unknown == 0
        assert res.status.free + res.status.occupied == 9
    assert correct >= 25, f"only {correct}/{total} slot states correct"


def test_tracker_hysteresis():
    tr = SlotTracker("s", k=5, min_frames=2)
    tr.update(1.0)
    assert tr.state == "unknown"          # not enough frames yet
    tr.update(1.0)
    assert tr.state == "occupied"
    tr.update(0.0)                        # one missed detection does not flip (mean 0.67)
    assert tr.state == "occupied"
    tr.update(0.0); tr.update(0.0)        # mean 0.4 -> free
    assert tr.state == "free"
    tr.update(None)                       # not observable keeps the state
    assert tr.state == "free"
