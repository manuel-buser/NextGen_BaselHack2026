"""Unit tests for the calibration helpers and the lattice row model (no detector needed)."""
from bikeslot.calibrate import (end_slots_from_bikes, fuse, marks_consistent, max_per_frame_in_row, row_model,
                                split_rows)
from bikeslot.geometry import pixel_polygon, point_in_polygon
from bikeslot.schemas import CalibrationHint
from bikeslot.stands import _fill_gaps, pitch_estimate

FRAME = (1280, 960)


def _bike(x, y, h=200.0, frame=0, far=None):
    return {"contact": [x, y], "far": far or [x, y - 0.8 * h], "w": 0.4 * h, "h": h, "frame": frame}


def _stand(x, y, w=30.0, h=60.0, conf=0.9):
    return {"x": x, "y": y, "w": w, "h": h, "conf": conf, "hits": 1, "inferred": False}


def _fuse(bikes, anchors, painted=None):
    frames = {b["frame"] for b in bikes} or {0}
    per_frame = [sum(b["frame"] == f for b in bikes) for f in sorted(frames)]
    return fuse("t", FRAME, bikes, anchors, None, [], painted, n_frames=len(frames),
                max_per_frame=max(per_frame), last_frame_vehicles=per_frame[-1])


def _slot_of(layout, point):
    hits = [s.index for s in layout.slots if point_in_polygon(pixel_polygon(s.polygon, FRAME), point)]
    assert len(hits) == 1, f"{point} is in slots {hits}"
    return hits[0]


# --------------------------------------------------------------------------- rows / helpers

def test_split_rows_two_parallel_diagonal_rows():
    front = [_bike(100 + i * 120, 600 + i * 30) for i in range(6)]          # diagonal row
    back = [_bike(300 + i * 100, 250 + i * 25, h=140) for i in range(8)]     # parallel row further up
    rows = split_rows(front + back)
    assert [len(r) for r in rows] == [6, 8]
    assert rows[0][0]["contact"][1] > rows[1][0]["contact"][1]               # front row first


def test_split_rows_single_curved_row_stays_one_row():
    bikes = [_bike(60 + i * 130, 400 + i * 18 + (i % 2) * 10) for i in range(9)]
    assert len(split_rows(bikes)) == 1


def test_max_per_frame_in_row_counts_per_frame():
    row = [_bike(0, 0, frame=0), _bike(1, 0, frame=0), _bike(2, 0, frame=1)]
    assert max_per_frame_in_row(row) == 2


def test_fill_gaps_infers_hidden_stands_and_caps_inference():
    anchors = [_stand(x, 300.0) for x in (100, 200, 300, 600, 700)]            # 400 and 500 hidden
    out = _fill_gaps(anchors)
    assert [round(a["x"]) for a in out] == [100, 200, 300, 400, 500, 600, 700]
    assert [a["inferred"] for a in out] == [False, False, False, True, True, False, False]
    assert abs(pitch_estimate([a["x"] for a in out]) - 100) < 1e-6


def test_end_slots_from_bikes():
    anchors = [_stand(x, 300.0) for x in (300, 400, 500, 600)]
    bikes = [_bike(80, 460), _bike(450, 460)]                 # one bike two pitches left of the first stand
    assert end_slots_from_bikes(bikes, anchors) == (2, 0)


def test_marks_consistent_rejects_plain_enumeration():
    pts = [[100, 0], [340, 0], [460, 0]]                      # gap of two pitches between bike 1 and 2
    hint = CalibrationHint(mark_slots={1: 1, 2: 2, 3: 3}, slot_count=5)
    ok, _ = marks_consistent(hint, pts, pitch=120)
    assert not ok
    hint = CalibrationHint(mark_slots={1: 1, 2: 3, 3: 4}, slot_count=5)
    ok, _ = marks_consistent(hint, pts, pitch=120)
    assert ok


# --------------------------------------------------------------------------- lattice row model

def _u(n, a=120.0, b=150.0, c=0.05):
    """The 9-bay perspective row: pitch 107 px at the near end, 60 px at the far end."""
    return (a * n + b) / (c * n + 1.0)


def test_perspective_row_with_hidden_stands_and_end_slot():
    seen = [0, 1, 3, 5, 7]                                    # stands 2, 4, 6 hidden behind bikes, 8 hidden too
    anchors = [_stand(_u(n), 300.0, w=25, h=50) for n in seen]
    parked = [2, 4, 6, 8]
    bikes = [_bike(_u(n) + 3.0, 520.0, h=220.0, far=[_u(n), 310.0]) for n in parked]
    layout = _fuse(bikes, anchors)
    assert len(layout.rows) == 1 and len(layout.slots) == 9
    assert "lattice from stands" in layout.calibration_hint.notes
    for n, b in zip(parked, bikes):
        assert _slot_of(layout, b["contact"]) == n + 1
    # slot polygons follow the perspective: the first slot is wider than the last one
    first, last = pixel_polygon(layout.slots[0].polygon, FRAME), pixel_polygon(layout.slots[-1].polygon, FRAME)
    assert (first[1][0] - first[0][0]) > 1.4 * (last[1][0] - last[0][0])


def test_stands_only_row():
    anchors = [_stand(200 + 100 * k, 400.0) for k in range(5)]
    layout = _fuse([], anchors)
    assert len(layout.slots) == 5
    assert all(len(s.polygon) == 4 for s in layout.slots)
    assert "slot depth from the stand height" in layout.calibration_hint.notes


def test_bikes_only_row_counts_the_empty_slot():
    bikes = [_bike(200 + 100 * k, 600.0) for k in (0, 1, 2, 4, 5)]
    layout = _fuse(bikes, [])
    assert len(layout.slots) == 6
    assert sorted(_slot_of(layout, b["contact"]) for b in bikes) == [1, 2, 3, 5, 6]


def test_single_bike_falls_back_to_affine_row():
    layout = _fuse([_bike(600, 600)], [])
    assert len(layout.slots) >= 1 and "affine fallback" in layout.calibration_hint.notes


def test_painted_dividers_outrank_off_centre_stands():
    # bays 130 px wide with dividers at 90 + 130 j; stands drawn every 110 px (an AI-image quirk)
    anchors = [_stand(100 + 110 * k, 300.0) for k in range(8)]
    dividers = [[90 + 130 * j, 300.0] for j in range(8)]
    bikes = [_bike(155 + 130 * j, 520.0, h=220.0, far=[155 + 130 * j, 310.0]) for j in range(7)]
    painted = {"back": dividers, "front": [[x, 540.0] for x, _ in dividers], "count": 7, "spacing_cv": 0.01}
    layout = _fuse(bikes, anchors, painted)
    assert len(layout.slots) == 7
    assert "lattice from dividers" in layout.calibration_hint.notes
    assert sorted(_slot_of(layout, b["contact"]) for b in bikes) == list(range(1, 8))


def test_stirrup_row_used_from_both_sides():
    # stirrups at y=400; bikes in front have their rear wheel 160 px below the stand line, the two bikes
    # parked on the far side have their near wheel in the stirrup (contact 20 px above the line) and extend
    # away from the camera
    anchors = [_stand(200 + 100 * k, 400.0) for k in range(6)]
    front = [_bike(200 + 100 * k + 5, 560.0, h=230.0, far=[200 + 100 * k, 330.0]) for k in (0, 2, 5)]
    behind = [_bike(200 + 100 * k - 5, 380.0, h=150.0, far=[200 + 100 * k, 230.0]) for k in (1, 4)]
    model = row_model(front + behind, anchors, None, FRAME)
    assert model is not None and model.bilateral and model.count == 6
    layout = _fuse(front + behind, anchors)
    assert len(layout.rows) == 2 and len(layout.slots) == 12
    assert "used from both sides" in layout.calibration_hint.notes
    assert len({_slot_of(layout, b["contact"]) for b in front + behind}) == 5


def test_full_row_behind_a_stand_row_stays_a_row_of_its_own():
    # two racks facing each other across a narrow aisle: the far row is as populated as the near one
    anchors = [_stand(200 + 100 * k, 400.0) for k in (1, 4)]
    front = [_bike(200 + 100 * k, 650.0, h=260.0, far=[200 + 100 * k, 410.0]) for k in range(6)]
    behind = [_bike(200 + 100 * k, 370.0, h=210.0, far=[200 + 100 * k, 170.0]) for k in range(6)]
    layout = _fuse(front + behind, anchors)
    assert "far side" not in layout.calibration_hint.notes
    assert len(layout.rows) == 2 and len(layout.slots) == 12
