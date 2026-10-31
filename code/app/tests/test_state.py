from bikeslot.config import Settings
from bikeslot.state import FREE, TAKEN, Rack

SETTINGS = Settings(free_after_clean=3, min_confidence=0.4)


def slots(*states: str, confidence: float = 0.9) -> dict:
    return {"slots": [{"id": i + 1, "state": s, "confidence": confidence} for i, s in enumerate(states)]}


def test_slot_mode_needs_three_clean_pictures_before_free():
    rack = Rack("A", "A", 2, window=3)
    rack.apply(slots(FREE, TAKEN), SETTINGS, now=1.0)
    rack.apply(slots(FREE, TAKEN), SETTINGS, now=2.0)
    assert rack.ready is False and rack.free is None and rack.status(SETTINGS, 2.0) == "starting"
    rack.apply(slots(FREE, TAKEN), SETTINGS, now=3.0)
    assert rack.ready is True and rack.free == 1 and rack.status(SETTINGS, 3.0) == "ok"


def test_slot_mode_one_sighting_makes_taken_at_once():
    rack = Rack("A", "A", 1, window=3)
    for t in range(3):
        rack.apply(slots(FREE), SETTINGS, now=float(t))
    assert rack.free == 1
    rack.apply(slots(TAKEN), SETTINGS, now=4.0)
    assert rack.free == 0
    rack.apply(slots(FREE), SETTINGS, now=5.0)
    rack.apply(slots(FREE), SETTINGS, now=6.0)
    assert rack.free == 0  # still inside the window with a sighting
    rack.apply(slots(FREE), SETTINGS, now=7.0)
    assert rack.free == 1


def test_weak_free_and_unsure_count_as_taken():
    rack = Rack("A", "A", 2, window=1)
    rack.apply({"slots": [{"id": 1, "state": FREE, "confidence": 0.2}, {"id": 2, "state": "unsure", "confidence": 0.9}]}, SETTINGS, 1.0)
    assert rack.free == 0


def test_count_mode_drops_at_once_and_rises_with_the_window():
    rack = Rack("A", "A", 10, window=3)
    for t in range(3):
        rack.apply({"bike_count": 4}, SETTINGS, now=float(t))
    assert rack.free == 6
    rack.apply({"bike_count": 7}, SETTINGS, now=3.0)
    assert rack.free == 3
    rack.apply({"bike_count": 2}, SETTINGS, now=4.0)
    rack.apply({"bike_count": 2}, SETTINGS, now=5.0)
    assert rack.free == 3
    rack.apply({"bike_count": 2}, SETTINGS, now=6.0)
    assert rack.free == 8


def test_stale_after_silence():
    settings = Settings(free_after_clean=1, stale_after_s=10)
    rack = Rack("A", "A", 1, window=1)
    rack.apply(slots(FREE), settings, now=100.0)
    assert rack.status(settings, 105.0) == "ok"
    assert rack.status(settings, 111.0) == "stale"
