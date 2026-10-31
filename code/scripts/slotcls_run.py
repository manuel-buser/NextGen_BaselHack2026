"""Run the slot classifier on one image or a folder of snapshots and write annotated overlays.

    .venv/Scripts/python code/scripts/slotcls_run.py --layout assets/pictures/DemoPictures/demo_picture.layout.json \
        --images assets/pictures/DemoPictures/demo_picture.jpeg --labels assets/pictures/DemoPictures/demo_picture.labels.json

A folder is processed in name order as a sequence (the temporal smoothing runs across the frames, like in
the service); with ``--no-smoothing`` every frame is judged on its own (k = min_frames = 1).  Outputs go to
``--out-dir`` (default data/slotcls/runs): ``<name>.overlay.jpg`` with the per-slot probability and colour,
``<name>.crops.jpg`` with what the classifier saw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config  # noqa: E402
from bikeslot.geometry import imread  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402
from bikeslot.slotcls import pipeline, smoothing  # noqa: E402
from bikeslot.slotcls.crops import contact_sheet  # noqa: E402
from bikeslot.slotcls.model import SlotClassifier  # noqa: E402
from slotcls_calibrate import draw_overlay  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--model", default=str(config.SLOTCLS_MODEL))
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or folders")
    ap.add_argument("--labels", nargs="*", default=[], help="label files to score against (by file name)")
    ap.add_argument("--out-dir", default=str(config.DATA_DIR / "slotcls" / "runs"))
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--no-blur", action="store_true", help="skip the person detector (faster, previews unblurred)")
    args = ap.parse_args()

    layout = Layout.model_validate_json(Path(args.layout).read_text(encoding="utf-8"))
    if args.no_smoothing:
        layout.params = {**(layout.params or {}), "k_frames": 1, "min_frames": 1}
    model = SlotClassifier.load(args.model)
    trackers = smoothing.make_trackers(layout)
    truth: dict[str, dict[str, str]] = {}
    for lf in args.labels:
        import json
        for fr in json.loads(Path(lf).read_text(encoding="utf-8")).get("frames", []):
            truth[Path(fr["file"]).name] = fr.get("slots", {})

    files: list[Path] = []
    for item in args.images:
        p = Path(item)
        files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct = total = 0
    for f in files:
        img = imread(f)
        if img is None:
            print(f"skip {f}: cannot read")
            continue
        res = pipeline.analyze_frame(layout, img, layout.camera_id, trackers, model=model, blur_persons=not args.no_blur)
        st = res.status
        states = {s.slot_id: s.status for s in st.slots}
        line = f"{f.name}: free {st.free} / occupied {st.occupied} / unknown {st.unknown}  ({st.processing_ms} ms)"
        want = truth.get(f.name)
        if want:
            ok = sum(states.get(k) == v for k, v in want.items())
            correct += ok
            total += len(want)
            line += f"  {ok}/{len(want)} vs labels"
            for k, v in want.items():
                if states.get(k) != v:
                    line += f"\n    {k}: predicted {states.get(k)} (p={res.probs.get(k, 0):.2f}), labelled {v}"
        print(line)
        print("  p(occupied): " + " ".join(f"{s.index}:{res.probs.get(s.slot_id, 0):.2f}" for s in layout.slots))
        overlay = draw_overlay(img, layout, states, res.probs)
        cv2.imwrite(str(out_dir / f"{f.stem}.overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out_dir / f"{f.stem}.crops.jpg"), contact_sheet(res.crops, states, res.probs, cols=9, scale=2))
    if total:
        print(f"\n{correct}/{total} slot states correct vs labels ({correct / total:.1%})")
    print(f"overlays written to {out_dir}")


if __name__ == "__main__":
    main()
