import numpy as np

from bikeslot import config
from bikeslot.geometry import (denormalize, interpolate_row, normalize, pixel_polygon, point_in_polygon,
                               polygon_iou, principal_axis_order, row_from_polygons, slot_masks)
from bikeslot.schemas import Layout, Row
from bikeslot.store import layout_from_golden


def test_normalize_roundtrip():
    pts = [[10, 20], [1270, 940]]
    assert denormalize(normalize(pts, [1280, 960]), [1280, 960]) == [[10.0, 20.0], [1270.0, 940.0]]


def test_interpolate_row_numbering_and_shape():
    row = Row(slot_count=4, front=[[0.0, 1.0], [1.0, 1.0]], back=[[0.0, 0.5], [1.0, 0.5]])
    slots = interpolate_row(row, "camX")
    assert [s.slot_id for s in slots] == ["camX-S01", "camX-S02", "camX-S03", "camX-S04"]
    assert slots[0].polygon == [[0.0, 0.5], [0.25, 0.5], [0.25, 1.0], [0.0, 1.0]]
    row.direction = "right_to_left"
    slots = interpolate_row(row, "camX")
    assert slots[0].slot_id == "camX-S01" and slots[0].polygon[0] == [0.75, 0.5]


def test_row_from_golden_polygons_matches_ends():
    gold = layout_from_golden()
    assert gold.status == "confirmed" and len(gold.slots) == 9
    row = row_from_polygons([s.polygon for s in gold.slots])
    regen = interpolate_row(row, gold.camera_id)
    # ends of the interpolated row coincide with the golden end slots, interior slots overlap well
    assert np.allclose(regen[0].polygon[0], gold.slots[0].polygon[0], atol=1e-5)
    assert np.allclose(regen[-1].polygon[1], gold.slots[-1].polygon[1], atol=1e-5)
    ious = [polygon_iou(a.polygon, b.polygon) for a, b in zip(regen, gold.slots)]
    assert min(ious) > 0.7 and sum(ious) / len(ious) > 0.8   # perspective-correct split: 0.76 / 0.85 measured


def test_masks_and_point_test():
    gold = layout_from_golden()
    masks = slot_masks(gold, (1280, 853))
    assert set(masks) == {s.slot_id for s in gold.slots}
    mask, poly = masks["cam01-S05"]
    centre = poly.mean(axis=0)
    assert point_in_polygon(poly, centre)
    assert mask[int(centre[1]), int(centre[0])] == 1
    assert not point_in_polygon(poly, (5, 5))


def test_polygon_iou_identity_and_disjoint():
    a = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert abs(polygon_iou(a, a) - 1.0) < 1e-6
    assert polygon_iou(a, [[2, 2], [3, 2], [3, 3], [2, 3]]) == 0.0


def test_principal_axis_order():
    pts = np.array([[3, 1.1], [1, 1.0], [2, 0.9]])
    assert list(principal_axis_order(pts)) == [1, 2, 0]
