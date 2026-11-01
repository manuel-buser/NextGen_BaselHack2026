"""Benchmark the automatic calibration (Stage A) on the golden scene and the generated scenes.

Usage (from the repo root):
    .venv/Scripts/python code/scripts/calib_bench.py [--what golden|gpt|all] [--vlm 0|1|both] [--previews DIR]

The raw proposal (no admin) is scored on the same image(s): golden per-slot states vs the hand labels (3 x 9 slots);
generated scenes have no labels, so "placed" = slots occupied vs bikes detected.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make "bikeslot" importable

import numpy as np  # noqa: E402

from bikeslot import config  # noqa: E402
from bikeslot.calibrate import propose_layout  # noqa: E402
from bikeslot.geometry import imread, polygon_iou, resize_max_side  # noqa: E402
from bikeslot.occupancy import SlotMaskCache, analyze_frame, make_trackers  # noqa: E402
from bikeslot.preview import render  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402
from bikeslot.store import layout_from_golden  # noqa: E402


class _Cam:
    """Duck-typed store.CameraState with the given frames buffered."""

    def __init__(self, frames, camera_id):
        self.camera_id = camera_id
        self.layout = Layout(camera_id=camera_id)
        self.frames = deque((None, f) for f in frames)


def _short(notes: str) -> str:
    return " | ".join(s for s in notes.split(" | ") if not s.startswith("[gpt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["golden", "gpt", "all"], default="all")
    ap.add_argument("--vlm", choices=["0", "1", "both"], default="0")
    ap.add_argument("--previews", default=str(config.DATA_DIR / "preview" / "bench"))
    ap.add_argument("--quiet", action="store_true", help="do not print the calibration notes")
    args = ap.parse_args()
    vlms = {"0": [False], "1": [True], "both": [False, True]}[args.vlm]
    out = Path(args.previews)
    out.mkdir(parents=True, exist_ok=True)

    if args.what in ("golden", "all"):
        gold = layout_from_golden()
        labels = json.load(open(config.GOLDEN_LABELS, encoding="utf-8"))["images"]
        names = [im["file"] for im in labels]
        truth = {im["file"]: im["slots"] for im in labels}
        frames = [resize_max_side(imread(config.GOLDEN_DIR / n), config.MAX_SIDE)[0] for n in names]
        for label, fr in [("all frames", frames)] + [(f"single {n[-7:-4]}", [f]) for n, f in zip(names, frames)]:
            for use_vlm in vlms:
                t = time.time()
                lay = propose_layout(_Cam(fr, "g"), use_vlm=use_vlm, use_stands=True)
                ms = round((time.time() - t) * 1000)
                lay.status = "confirmed"
                lay.params = {"k_frames": 1, "min_frames": 1}
                ok = tot = 0
                res = None
                for n, f in zip(names, frames):
                    res = analyze_frame(lay, make_trackers(lay), SlotMaskCache(), f, "g")
                    pred = {s.index: s.status for s in res.status.slots}
                    for sid, want in truth[n].items():
                        tot += 1
                        ok += pred.get(int(sid[-2:])) == want
                iou = ""
                if len(lay.slots) == len(gold.slots):
                    vals = [polygon_iou(p.polygon, q.polygon) for p, q in zip(lay.slots, gold.slots)]
                    iou = f", IoU vs hand-labelled bays mean {np.mean(vals):.2f} min {min(vals):.2f}"
                print(f"GOLDEN {label:>11} vlm={int(use_vlm)} ({ms} ms): N={len(lay.slots)} occupancy {ok}/{tot}{iou}")
                if not args.quiet:
                    print("   ", _short(lay.calibration_hint.notes))
                tag = label.replace(" ", "_")
                (out / f"golden_{tag}_vlm{int(use_vlm)}.jpg").write_bytes(
                    render(res.frame, res.status, res.detections, res.assignments))

    if args.what in ("gpt", "all"):
        folder = config.ROOT / "assets" / "pictures" / "GeneratedByChatGPT"
        for p in sorted(folder.glob("bikeparking_gpt_*.png")):
            f, _ = resize_max_side(imread(p), config.MAX_SIDE)
            for use_vlm in vlms:
                t = time.time()
                try:
                    lay = propose_layout(_Cam([f], p.stem), use_vlm=use_vlm, use_stands=True)
                except ValueError as exc:
                    print(f"GPT {p.name} vlm={int(use_vlm)}: FAILED {exc}")
                    continue
                ms = round((time.time() - t) * 1000)
                lay.status = "confirmed"
                lay.params = {"k_frames": 1, "min_frames": 1}
                res = analyze_frame(lay, make_trackers(lay), SlotMaskCache(), f, p.stem)
                (out / f"{p.stem}_vlm{int(use_vlm)}.jpg").write_bytes(
                    render(res.frame, res.status, res.detections, res.assignments))
                bikes = res.status.detections.get("bicycle", 0) + res.status.detections.get("motorcycle", 0)
                print(f"GPT {p.name} vlm={int(use_vlm)} ({ms} ms): N={len(lay.slots)} rows={len(lay.rows)} "
                      f"type={lay.parking_type} -> placed {res.status.occupied}/{bikes} bikes")
                if not args.quiet:
                    print("   ", _short(lay.calibration_hint.notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
