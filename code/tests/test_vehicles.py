"""Vehicle detectors: switching by name / configuration, and Grounding DINO on a real picture when its weights are cached."""
import cv2
import numpy as np
import pytest

from bikeslot import config, vehicles
from bikeslot.schemas import Detection


@vehicles.register("fake")
class FakeVehicles(vehicles.VehicleDetector):
    def load(self):
        self._info["loaded"] = True

    def detect(self, img):
        return [Detection(cls=config.CLS_BICYCLE, name="bicycle", conf=0.9, box=[1.0, 2.0, 3.0, 4.0], contact=[2.0, 4.0])]


def test_registry_and_switching(monkeypatch):
    assert {"yolo", "dino", "fake"} <= set(vehicles.available())
    monkeypatch.setattr(config, "VEHICLE_DETECTOR", "fake")
    dets = vehicles.detect(np.zeros((10, 10, 3), np.uint8))
    assert dets[0].name == "bicycle" and vehicles.get() is vehicles.get("fake")
    assert vehicles.info()["selected"] == "fake"
    with pytest.raises(ValueError, match="available"):
        vehicles.get("nope")


def _det(box):
    return Detection(cls=config.CLS_BICYCLE, name="bicycle", conf=0.5, box=box, contact=[(box[0] + box[2]) / 2, box[3]])


def test_one_vehicle_twice_keeps_the_outer_box():
    loose, tight, apart = _det([0, 0, 100, 210]), _det([20, 40, 60, 199]), _det([300, 0, 360, 120])
    out = vehicles.drop_containers([loose, tight, apart])
    assert out == [loose, apart]
    half_out = _det([80, 0, 140, 120])                     # overlaps the big box but is not inside it
    assert vehicles.drop_containers([loose, half_out]) == [loose, half_out]


def test_wheel_and_basket_inside_a_bike_count_once():
    bike, wheel, basket = _det([0, 0, 100, 200]), _det([10, 120, 60, 200]), _det([30, 40, 90, 90])
    out = vehicles.drop_containers([bike, wheel, basket])
    assert len(out) == 1 and out[0].contact[1] == 200


def test_box_around_bikes_side_by_side_is_dropped():
    group, left, right = _det([0, 0, 300, 200]), _det([10, 10, 140, 199]), _det([160, 5, 290, 198])
    assert vehicles.drop_containers([group, left, right]) == [left, right]


def test_rack_area_is_widened_by_a_hook_at_each_end():
    from bikeslot.schemas import Layout, Row

    lay = Layout(camera_id="c", rows=[Row(slot_count=10, back=[[0.2, 0.5], [0.7, 0.5]], front=[[0.2, 0.6], [0.7, 0.6]])])
    area = vehicles.rack_area(lay, (1000, 1000))
    assert area[0].tolist() == pytest.approx([150, 500]) and area[1].tolist() == pytest.approx([750, 500])
    assert area[2][1] == pytest.approx(500 + 2.2 * 100)
    assert vehicles.in_area(_det([140, 600, 160, 700]), area) and not vehicles.in_area(_det([100, 600, 120, 700]), area)


def _dino_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(config.DINO_MODEL, "config.json"), str)


@pytest.mark.skipif(not _dino_cached(), reason="Grounding DINO weights not downloaded")
def test_dino_finds_the_demo_bikes():
    img = cv2.imread(str(config.ROOT / "assets" / "pictures" / "DemoPictures" / "demo_picture.jpeg"))
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
    dets = vehicles.get("dino").detect(img)
    assert len(dets) >= 4                       # five bikes in the rack, two of them leaning
    assert all(d.contact and d.contact[1] == d.box[3] for d in dets)
