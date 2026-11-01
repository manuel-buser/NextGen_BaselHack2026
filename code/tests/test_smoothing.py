"""Per-camera smoothing (layout.params): occupied at once, free only after two free pictures in a row."""
from bikeslot.schemas import Layout, Slot
from bikeslot.slotcls import smoothing


def tracker(**params):
    layout = Layout(camera_id="c", slots=[Slot(slot_id="c-S01", index=1, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])],
                    params=params)
    return smoothing.make_trackers(layout)["c-S01"]


def states(tr, probs):
    out = []
    for p in probs:
        smoothing.update({"c-S01": tr}, {"c-S01": p})
        out.append(tr.state)
    return out


FAST = dict(k_frames=2, min_frames=1, occupied_above=0.5, free_below=0.0)


def test_one_occupied_picture_makes_the_hook_taken():
    assert states(tracker(**FAST), [0.0, 0.0, 1.0]) == ["free", "free", "occupied"]


def test_free_needs_two_free_pictures_in_a_row():
    assert states(tracker(**FAST), [1.0, 0.0, 1.0, 0.0, 0.0]) == ["occupied"] * 4 + ["free"]


def test_unsure_and_unseen_pictures_keep_the_state():
    assert states(tracker(**FAST), [1.0, 0.5, None, 0.0, 0.0]) == ["occupied"] * 4 + ["free"]


def test_the_service_trackers_follow_the_same_params():
    from bikeslot import occupancy                                   # store.CameraState builds its trackers here
    layout = Layout(camera_id="c", slots=[Slot(slot_id="c-S01", index=1, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])],
                    params=FAST)
    assert states(occupancy.make_trackers(layout)["c-S01"], [0.0, 1.0, 0.0, 0.0]) == ["free", "occupied", "occupied", "free"]


def test_without_params_the_default_majority_of_five_stays():
    tr = tracker()
    assert states(tr, [0.0] * 5 + [1.0, 1.0, 1.0]) == ["unknown"] + ["free"] * 6 + ["occupied"]
