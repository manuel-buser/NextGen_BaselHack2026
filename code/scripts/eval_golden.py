"""Evaluate the occupancy pipeline on the golden data set.

Usage (from the repo root):
    .venv/Scripts/python code/scripts/eval_golden.py [--labels path] [--previews DIR] [--min-correct 25]

Each image is analysed with fresh, single-frame trackers (k_frames=1, min_frames=1) so the result is the
pure per-frame decision; exit code 1 if fewer than --min-correct slot states are right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make "bikeslot" importable

import cv2  # noqa: E402

from bikeslot import config  # noqa: E402
from bikeslot.geometry import imread  # noqa: E402
from bikeslot.occupancy import SlotMaskCache, analyze_frame, make_trackers  # noqa: E402
from bikeslot.preview import render  # noqa: E402
from bikeslot.store import golden_truth, layout_from_golden  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(config.GOLDEN_LABELS))
    ap.add_argument("--previews", default=None, help="write annotated previews into this directory")
    ap.add_argument("--min-correct", type=int, default=25)
    args = ap.parse_args()

    labels = Path(args.labels)
    layout = layout_from_golden(labels)
    layout.params = {"k_frames": 1, "min_frames": 1}
    truth = golden_truth(labels)
    out_dir = Path(args.previews) if args.previews else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = correct = 0
    print(f"model={config.MODEL_NAME} imgsz={config.IMGSZ} conf={config.CONF}")
    for item in truth:
        img = imread(str(labels.parent / item["file"]))
        if img is None:
            print(f"!! cannot read {item['file']}")
            continue
        res = analyze_frame(layout, make_trackers(layout), SlotMaskCache(), img, layout.camera_id)
        pred = {s.slot_id: s.status for s in res.status.slots}
        rows = []
        img_ok = 0
        for sid, want in item["slots"].items():
            got = pred.get(sid, "missing")
            ok = got == want
            total += 1
            correct += ok
            img_ok += ok
            rows.append(f"{sid[-3:]}:{got[:3]}{'' if ok else '!!'}")
        print(f"{item['file']}  {img_ok}/{len(item['slots'])}  {res.status.processing_ms:.0f} ms  "
              f"dets={res.status.detections}  " + " ".join(rows))
        if out_dir:
            (out_dir / item["file"].replace(".png", ".jpg")).write_bytes(render(res.frame, res.status, res.detections, res.assignments))
    print(f"\nSLOT ACCURACY {correct}/{total}")
    return 0 if correct >= min(args.min_correct, total) else 1


if __name__ == "__main__":
    sys.exit(main())
