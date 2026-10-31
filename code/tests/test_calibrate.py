"""Unit tests for the calibration helpers that need no model."""
import numpy as np

from bikeslot.calibrate import end_slots_from_bikes, marks_consistent, max_per_frame_in_row, split_rows
from bikeslot.schemas import CalibrationHint
from bikeslot.stands import _fill_gaps, pitch_estimate


def _bike(x, y, h=200.0, frame=0):
    return {"contact": [x, y], "far": [x, y - 0.8 * h], "w": 0.4 * h, "h": h, "frame": frame}


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
    anchors = [{"x": x, "y": 300.0, "w": 30.0, "h": 60.0, "conf": 0.9, "hits": 1, "inferred": False}
               for x in (100, 200, 300, 600, 700)]            # 400 and 500 hidden
    out = _fill_gaps(anchors)
    assert [round(a["x"]) for a in out] == [100, 200, 300, 400, 500, 600, 700]
    assert [a["inferred"] for a in out] == [False, False, False, True, True, False, False]
    assert abs(pitch_estimate([a["x"] for a in out]) - 100) < 1e-6


def test_end_slots_from_bikes():
    anchors = [{"x": x, "y": 300.0, "w": 30.0, "h": 60.0, "conf": 0.9, "hits": 1, "inferred": False}
               for x in (300, 400, 500, 600)]
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
