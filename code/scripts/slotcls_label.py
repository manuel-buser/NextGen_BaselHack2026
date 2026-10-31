"""Label slots frame by frame: one click per slot toggles free <-> occupied.

    .venv/Scripts/python code/scripts/slotcls_label.py --layout assets/pictures/DemoPictures/demo_picture.layout.json \
        --images data/snapshots/demo01 --out data/slotcls/demo01.labels.json [--model assets/models/slotcls_demo.pth]

Window: the frame with all slot polygons (green = free, red = occupied, grey = unknown) and, on the right,
the enlarged crop of the slot under the mouse.  Every frame starts from the previous frame's labels (or the
model's prediction with --model), so usually only the slots that changed need a click.

    left click on a slot   toggle free / occupied        u + click   mark unknown (excluded from training)
    Enter / n              save and next frame           b           previous frame
    a                      all free                      p           fill from the model (--model)
    q / Esc                save and quit

The output is the label file format read by slotcls_train.py; frames already in the file are shown first
with their stored labels so a session can be resumed.  Image paths are stored relative to the label file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config  # noqa: E402
from bikeslot.geometry import denormalize, imread  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402
from bikeslot.slotcls.crops import slot_crops  # noqa: E402
from bikeslot.slotcls.dataset import LabelFile  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {"free": (80, 200, 80), "occupied": (60, 60, 230), "unknown": (160, 160, 160)}
WINDOW = "slotcls label"


def relpath(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return str(path.resolve())


class Labeller:
    def __init__(self, layout: Layout, label_file: LabelFile, files: list[Path], model=None):
        self.layout, self.lf, self.files, self.model = layout, label_file, files, model
        self.i = 0
        self.img = None
        self.scale = 1.0
        self.labels: dict[str, str] = {}
        self.hover: str | None = None
        self.mark_unknown = False
        self.polys_px: dict[str, np.ndarray] = {}
        self.crops: dict[str, np.ndarray] = {}
        self.view_w = 1400

    # ------------------------------------------------------------------ frame handling
    def stored(self, f: Path) -> dict | None:
        rel = relpath(f, self.lf.path.parent)
        return next((fr for fr in self.lf.frames if fr["file"] == rel), None)

    def load(self, i: int, previous: dict[str, str] | None = None) -> bool:
        f = self.files[i]
        img = imread(f)
        if img is None:
            print(f"cannot read {f}")
            return False
        self.i, self.img = i, img
        fs = (img.shape[1], img.shape[0])
        self.polys_px = {s.slot_id: np.array(denormalize(s.polygon, fs), np.float32) for s in self.layout.slots}
        self.crops = slot_crops(self.layout, img, size=(160, 240), margin=config.SLOTCLS_MARGIN)
        self.scale = min(1.0, self.view_w / img.shape[1])
        stored = self.stored(f)
        if stored:
            self.labels = {s.slot_id: stored["slots"].get(s.slot_id, "unknown") for s in self.layout.slots}
        elif self.model is not None:
            self.fill_from_model()
        elif previous:
            self.labels = dict(previous)
        else:
            self.labels = {s.slot_id: "free" for s in self.layout.slots}
        return True

    def fill_from_model(self) -> None:
        crops = slot_crops(self.layout, self.img, size=self.model.input_size, margin=config.SLOTCLS_MARGIN)
        ids = list(crops)
        probs = self.model.predict_proba([crops[s] for s in ids])
        self.labels = {sid: ("occupied" if p >= 0.5 else "free") for sid, p in zip(ids, probs)}

    def save(self) -> None:
        rel = relpath(self.files[self.i], self.lf.path.parent)
        entry = {"file": rel, "slots": dict(self.labels)}
        for k, fr in enumerate(self.lf.frames):
            if fr["file"] == rel:
                self.lf.frames[k] = entry
                break
        else:
            self.lf.frames.append(entry)
        self.lf.save()

    # ------------------------------------------------------------------ drawing
    def slot_at(self, x: float, y: float) -> str | None:
        for sid, poly in self.polys_px.items():
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                return sid
        return None

    def render(self) -> np.ndarray:
        img = cv2.resize(self.img, None, fx=self.scale, fy=self.scale) if self.scale < 1.0 else self.img.copy()
        overlay = img.copy()
        for s in self.layout.slots:
            poly = (self.polys_px[s.slot_id] * self.scale).round().astype(np.int32)
            color = COLORS[self.labels.get(s.slot_id, "unknown")]
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        for s in self.layout.slots:
            poly = (self.polys_px[s.slot_id] * self.scale).round().astype(np.int32)
            color = COLORS[self.labels.get(s.slot_id, "unknown")]
            cv2.polylines(img, [poly], True, (255, 255, 255) if s.slot_id == self.hover else color, 2 if s.slot_id == self.hover else 1)
            cx, cy = poly.mean(axis=0).astype(int)
            cv2.putText(img, str(s.index), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, str(s.index), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        n_occ = sum(v == "occupied" for v in self.labels.values())
        n_unk = sum(v == "unknown" for v in self.labels.values())
        head = (f"[{self.i + 1}/{len(self.files)}] {self.files[self.i].name}   occupied {n_occ}  free "
                f"{len(self.labels) - n_occ - n_unk}  unknown {n_unk}   click=toggle  u+click=unknown  Enter=next  b=back  q=quit"
                + ("   [UNKNOWN MODE]" if self.mark_unknown else ""))
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, head, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        side = np.full((img.shape[0], 180, 3), 40, np.uint8)
        if self.hover and self.hover in self.crops:
            c = self.crops[self.hover]
            side[30:30 + c.shape[0], 10:10 + c.shape[1]] = c
            cv2.putText(side, f"{self.hover.split('-')[-1]}: {self.labels.get(self.hover, '?')}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[self.labels.get(self.hover, "unknown")], 1)
        return np.hstack([img, side])

    def on_mouse(self, event, x, y, flags, param):
        sid = self.slot_at(x / self.scale, y / self.scale)
        if event == cv2.EVENT_MOUSEMOVE:
            if sid != self.hover:
                self.hover = sid
                self.dirty = True
        elif event == cv2.EVENT_LBUTTONDOWN and sid:
            if self.mark_unknown:
                self.labels[sid] = "unknown"
            else:
                self.labels[sid] = "free" if self.labels.get(sid) == "occupied" else "occupied"
            self.dirty = True

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        if not self.load(0):
            return
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.dirty = True
        while True:
            if self.dirty:
                cv2.imshow(WINDOW, self.render())
                self.dirty = False
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                self.save()
                break
            if key in (13, 10, ord("n")):
                self.save()
                if self.i + 1 >= len(self.files):
                    print("last frame labelled")
                    break
                prev = dict(self.labels)
                self.load(self.i + 1, previous=prev)
            elif key == ord("b") and self.i > 0:
                self.save()
                self.load(self.i - 1)
            elif key == ord("a"):
                self.labels = {s.slot_id: "free" for s in self.layout.slots}
            elif key == ord("p") and self.model is not None:
                self.fill_from_model()
            elif key == ord("u"):
                self.mark_unknown = not self.mark_unknown
            self.dirty = True
        cv2.destroyAllWindows()
        print(f"{len(self.lf.frames)} frame(s) in {self.lf.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True, help="layout JSON with the slot polygons (slotcls_calibrate.py)")
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or folders of snapshots")
    ap.add_argument("--out", required=True, help="label JSON to create or extend")
    ap.add_argument("--model", help="classifier used to pre-fill the labels of new frames")
    args = ap.parse_args()

    layout_path = Path(args.layout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        lf = LabelFile.load(out)
    else:
        layout = Layout.model_validate_json(layout_path.read_text(encoding="utf-8"))
        lf = LabelFile(path=out, camera_id=layout.camera_id, layout=layout, data_layout=relpath(layout_path, out.parent))
        lf.save()
    files: list[Path] = []
    for item in args.images:
        p = Path(item)
        files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p])
    if not files:
        sys.exit("no images found")
    done = {fr["file"] for fr in lf.frames}
    files.sort(key=lambda f: (relpath(f, out.parent) not in done, str(f)))   # already labelled first (resume)
    model = None
    if args.model:
        from bikeslot.slotcls.model import SlotClassifier
        model = SlotClassifier.load(args.model)
    Labeller(lf.layout, lf, files, model).run()


if __name__ == "__main__":
    main()
