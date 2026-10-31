"""Calibrate once: define the slot polygons of a fixed camera and store them as a layout JSON.

The quickest way for a straight rack with equally spaced slots is 4 corners + the slot count: the row
quadrilateral is split into N equal columns *in perspective* (homography), exactly like the admin editor.
Corner order: top-left, top-right, bottom-right, bottom-left, in pixels of the given image.

Examples (from the repo root):
    # the 19 slots of bay 2 between its two pillars in the demo picture, from 4 corners
    .venv/Scripts/python code/scripts/slotcls_calibrate.py --image assets/pictures/DemoPictures/demo_picture.jpeg
        --camera demo01 --slots 18 --corners 248,770 1462,910 1498,1135 128,995
        --out assets/pictures/DemoPictures/demo_picture.layout.json

    # same, but click the 4 corners in a window (left click = corner, right click = undo, Enter = done)
    .venv/Scripts/python code/scripts/slotcls_calibrate.py --image ... --camera demo01 --slots 18 --interactive --out ...

    # only redraw the overlay + crop sheet of an existing layout (e.g. after editing polygons by hand)
    .venv/Scripts/python code/scripts/slotcls_calibrate.py --image ... --layout assets/pictures/DemoPictures/demo_picture.layout.json

Outputs next to --out: ``<name>.overlay.jpg`` (polygons drawn on the image) and ``<name>.crops.jpg``
(what the classifier will see).  Individual polygons can then be edited in the JSON (normalised 0..1) or
through ``PUT /cameras/{id}/layout`` in the admin UI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config  # noqa: E402
from bikeslot.geometry import denormalize, imread, interpolate_row, normalize  # noqa: E402
from bikeslot.schemas import Layout, Row  # noqa: E402
from bikeslot.slotcls.crops import contact_sheet, slot_crops  # noqa: E402

WINDOW = "slotcls calibrate"


def parse_corners(items: list[str]) -> list[list[float]]:
    pts = []
    for it in items:
        x, y = it.replace(";", ",").split(",")
        pts.append([float(x), float(y)])
    if len(pts) != 4:
        sys.exit("--corners needs exactly 4 points: top-left top-right bottom-right bottom-left")
    return pts


def pick_corners(img: np.ndarray, n_slots: int) -> list[list[float]]:
    """Tiny OpenCV UI: click top-left, top-right, bottom-right, bottom-left; right click undoes; Enter accepts."""
    scale = min(1.0, 1500.0 / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
    pts: list[list[float]] = []
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def redraw():
        canvas = view.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x * scale), int(y * scale)), 5, (0, 255, 255), -1)
            cv2.putText(canvas, names[i], (int(x * scale) + 6, int(y * scale) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(pts) == 4:
            row = Row(slot_count=n_slots, back=[pts[0], pts[1]], front=[pts[3], pts[2]])
            for s in interpolate_row(row, "tmp"):
                poly = (np.array(s.polygon, np.float32) * scale).round().astype(np.int32)
                cv2.polylines(canvas, [poly], True, (0, 200, 0), 1)
        msg = f"click {names[len(pts)]}" if len(pts) < 4 else "Enter = accept, right click = undo"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(canvas, msg, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x / scale, y / scale])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(pts) == 4:
            break
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("aborted")
    cv2.destroyAllWindows()
    return pts


def layout_from_corners(camera_id: str, frame_size: tuple[int, int], corners_px: list[list[float]], n_slots: int,
                        site_id: str = "site01", site_name: str = "", direction: str = "left_to_right") -> Layout:
    """Row quad (top-left, top-right, bottom-right, bottom-left in pixels) + N -> confirmed classifier layout."""
    tl, tr, br, bl = corners_px
    row = Row(row_id="r1", slot_count=n_slots, back=normalize([tl, tr], frame_size),
              front=normalize([bl, br], frame_size), direction=direction)
    return Layout(camera_id=camera_id, site_id=site_id, site_name=site_name, frame_size=list(frame_size),
                  parking_type="wheel_holder_rack", status="confirmed", source="admin", occupancy_mode="per_slot",
                  rows=[row], slots=interpolate_row(row, camera_id), params={"engine": "classifier"})


def draw_overlay(img: np.ndarray, layout: Layout, labels: dict[str, str] | None = None,
                 probs: dict[str, float] | None = None) -> np.ndarray:
    out = img.copy()
    fs = (img.shape[1], img.shape[0])
    colors = {"free": (80, 200, 80), "occupied": (60, 60, 230), "unknown": (160, 160, 160)}
    for s in layout.slots:
        poly = np.array(denormalize(s.polygon, fs), np.float32).round().astype(np.int32)
        color = colors.get(labels.get(s.slot_id, "") if labels else "", (0, 220, 220))
        cv2.polylines(out, [poly], True, color, 2)
        cx, cy = poly.mean(axis=0).astype(int)
        txt = str(s.index) + (f" {probs[s.slot_id]:.2f}" if probs and s.slot_id in probs else "")
        cv2.putText(out, txt, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, txt, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def output_stem(layout_path: Path) -> Path:
    stem = str(layout_path.with_suffix(""))
    return Path(stem[:-len(".layout")] if stem.endswith(".layout") else stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="camera image the polygons are drawn on")
    ap.add_argument("--camera", default="demo01")
    ap.add_argument("--site", default="site01")
    ap.add_argument("--site-name", default="")
    ap.add_argument("--slots", type=int, default=18, help="number of equally spaced slots between the corners")
    ap.add_argument("--corners", nargs=4, metavar="X,Y", help="top-left top-right bottom-right bottom-left (pixels)")
    ap.add_argument("--direction", choices=["left_to_right", "right_to_left"], default="left_to_right")
    ap.add_argument("--interactive", action="store_true", help="click the 4 corners in a window")
    ap.add_argument("--layout", help="existing layout JSON: only render overlay/crops (no new layout)")
    ap.add_argument("--out", help="layout JSON to write (default: <image>.layout.json)")
    ap.add_argument("--crop-size", type=int, default=config.SLOTCLS_INPUT)
    ap.add_argument("--margin", type=float, default=config.SLOTCLS_MARGIN)
    args = ap.parse_args()

    img = imread(args.image)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    frame_size = (img.shape[1], img.shape[0])

    if args.layout:
        layout = Layout.model_validate_json(Path(args.layout).read_text(encoding="utf-8"))
        out_path = Path(args.layout)
    else:
        if args.interactive:
            corners = pick_corners(img, args.slots)
        elif args.corners:
            corners = parse_corners(args.corners)
        else:
            sys.exit("give --corners, --interactive or --layout")
        layout = layout_from_corners(args.camera, frame_size, corners, args.slots, args.site, args.site_name, args.direction)
        out_path = Path(args.out) if args.out else Path(args.image).with_suffix(".layout.json")
        out_path.write_text(layout.model_dump_json(indent=2), encoding="utf-8")
        print(f"layout with {len(layout.slots)} slots written to {out_path}")

    stem = output_stem(out_path)
    cv2.imwrite(f"{stem}.overlay.jpg", draw_overlay(img, layout), [cv2.IMWRITE_JPEG_QUALITY, 88])
    crops = slot_crops(layout, img, size=args.crop_size, margin=args.margin)
    cv2.imwrite(f"{stem}.crops.jpg", contact_sheet(crops, cols=9, scale=2), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"overlay: {stem}.overlay.jpg   crops: {stem}.crops.jpg")


if __name__ == "__main__":
    main()
