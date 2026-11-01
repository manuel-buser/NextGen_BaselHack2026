"""Hook allocation: hooks that can only be spill-over of the vehicles seen are freed, nothing else."""
import sys
from pathlib import Path

import numpy as np

from bikeslot import config, vehicles
from bikeslot.geometry import denormalize
from bikeslot.schemas import Detection
from bikeslot.slotcls import pipeline, smoothing
from bikeslot.slotcls.allocate import FREE, TAKEN, clusters_of, decide, hook_positions, possible_bike_hooks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from slotcls_calibrate import layout_from_corners  # noqa: E402


def probs(**high):
    p = {i: 0.02 for i in range(1, 20)}
    p.update({int(k[1:]): v for k, v in high.items()})
    return p


def freed(a):
    assert TAKEN not in a.decisions.values()                         # the step never makes a hook taken
    return sorted(i for i, d in a.decisions.items() if d == FREE)


def test_clusters_are_separated_by_clearly_free_hooks():
    assert clusters_of(probs(h2=0.9, h3=0.5, h5=0.8), 0.15) == [[2, 3], [5]]


def test_possible_bike_hooks_light_the_whole_cluster():
    assert possible_bike_hooks([1, 2, 3], 1, 1) == [2]
    assert possible_bike_hooks([1, 2], 1, 1) == [1, 2]
    assert possible_bike_hooks([4, 5, 6, 7, 8], 2, 1) == [4, 5, 7, 8]
    assert possible_bike_hooks([1, 2, 3, 4], 1, 1) == []             # one bike cannot light four hooks
    assert possible_bike_hooks([1, 2, 3], 3, 1) == [1, 2, 3]


def test_spill_over_around_one_bike_is_freed():
    a = decide(probs(h1=0.95, h2=1.0, h3=0.8), [(1.9, 2.5)])
    assert freed(a) == [1, 3] and 2 not in a.decisions               # the classifier keeps hook 2 taken


def test_box_position_does_not_matter_only_the_count():
    for box in [(2.3, 2.9), (1.9, 3.8), (-1.0, 6.0)]:                 # narrow, leaning, wide box over the same bike
        assert freed(decide(probs(h1=0.97, h2=1.0, h3=0.97), [box])) == [1, 3]


def test_two_lit_hooks_with_one_bike_both_stay():
    a = decide(probs(h13=1.0, h14=0.97), [(10.0, 16.0)])            # either hook can be the bike's
    assert freed(a) == []


def test_two_separate_bikes_free_their_echoes():
    a = decide(probs(h10=0.9, h17=0.7, h18=1.0, h19=0.9), [(9.8, 10.4), (17.2, 18.4)])
    assert freed(a) == [17, 19]


def test_packed_bikes_in_one_box_free_nothing():
    a = decide(probs(h15=0.9, h16=1.0, h17=0.95, h18=0.9, h19=0.8), [(14.5, 19.5)])   # memox_07: 3 bikes, 1 box
    assert freed(a) == []


def test_hook_between_two_bikes_is_not_freed():
    a = decide(probs(h4=0.9, h5=0.97, h6=0.85), [(3.7, 4.3), (5.7, 6.3)])   # could be a third bike
    assert freed(a) == []


def test_three_side_by_side_with_three_vehicles_free_nothing():
    a = decide(probs(h5=0.9, h6=0.95, h7=0.9, h8=0.4), [(4.7, 5.3), (5.8, 6.4), (6.9, 7.5)])
    assert freed(a) == []


def test_no_vehicle_seen_frees_nothing():
    a = decide(probs(h3=0.95, h4=0.5, h12=0.45, h13=0.3), [])
    assert freed(a) == [] and a.decisions == {}


def test_vehicles_over_no_cluster_change_nothing():
    a = decide(probs(h7=0.9), [(6.7, 7.3), (14.8, 15.2), (15.3, 15.7)])   # e.g. a rod counted twice
    assert a.decisions == {} and a.clusters == [{"hooks": [7], "vehicles": 1, "possible": [7]}]


def test_hook_positions_interpolate_and_extrapolate():
    hx = {1: 100.0, 2: 120.0, 3: 140.0}
    assert hook_positions([110.0, 140.0, 80.0, 160.0], hx) == [1.5, 3.0, 0.0, 4.0]


class StubModel:
    input_size = (96, 96)

    def predict_proba(self, crops):                     # hooks 3, 4, 5 light up (one bike on 4 plus spill-over)
        return np.array([{2: 0.8, 3: 0.99, 4: 0.7}.get(i, 0.02) for i in range(len(crops))], np.float32)


@vehicles.register("one-bike-on-hook-4")
class OneBike(vehicles.VehicleDetector):
    layout = None

    def load(self):
        pass

    def detect(self, img):
        s = next(s for s in self.layout.slots if s.index == 4)
        pts = denormalize(s.polygon, (img.shape[1], img.shape[0]))
        x = float(np.mean([p[0] for p in pts])); y = float(max(p[1] for p in pts))
        return [Detection(cls=config.CLS_BICYCLE, name="bicycle", conf=0.5, box=[x - 15, y - 120, x + 15, y], contact=[x, y])]


def test_pipeline_lets_the_detected_bike_claim_its_hook(monkeypatch):
    layout = layout_from_corners("c", (640, 480), [[60, 200], [580, 200], [580, 300], [60, 300]], 10)
    layout.params.update(allocation="dino", k_frames=1, min_frames=1)
    OneBike.layout = layout
    monkeypatch.setattr(config, "ALLOC_DETECTOR", "one-bike-on-hook-4")
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)   # sharp enough to vote
    res = pipeline.analyze_frame(layout, img, "c", smoothing.make_trackers(layout), model=StubModel(), blur_persons=False)
    st = res.status
    assert [s.index for s in st.slots if s.status == "occupied"] == [4]
    assert st.allocation["vehicles"] == 1 and st.allocation["freed"] == [3, 5]
    assert st.allocation["clusters"] == [{"hooks": [3, 4, 5], "vehicles": 1, "possible": [4]}]
    assert any(d.name == "bicycle" for d in res.detections)       # drawn on the preview

    layout.params.pop("allocation")                                   # switched off: the classifier alone
    res = pipeline.analyze_frame(layout, img, "c", smoothing.make_trackers(layout), model=StubModel(), blur_persons=False)
    assert [s.index for s in res.status.slots if s.status == "occupied"] == [3, 4, 5] and res.status.allocation is None


def test_stage_pictures_show_each_model_on_its_own(monkeypatch):
    import cv2

    from bikeslot.preview import stages

    layout = layout_from_corners("c", (640, 480), [[60, 200], [580, 200], [580, 300], [60, 300]], 10)
    layout.params.update(allocation="dino", k_frames=1, min_frames=1)
    OneBike.layout = layout
    monkeypatch.setattr(config, "ALLOC_DETECTOR", "one-bike-on-hook-4")
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    res = pipeline.analyze_frame(layout, img, "c", smoothing.make_trackers(layout), model=StubModel(), blur_persons=False)
    jpegs, counts = stages(res.frame, res.status, res.detections, res.probs)
    assert all(cv2.imdecode(np.frombuffer(jpegs[k], np.uint8), cv2.IMREAD_COLOR) is not None for k in ("camera", "hooks", "bikes"))
    assert counts["hooks"]["occupied"] == 3 and counts["bikes"]["occupied"] == 1 and counts["bikes"]["vehicles"] == 1
    assert res.timings["classifier_ms"] >= 0 and res.status.allocation["ms"] >= 0
