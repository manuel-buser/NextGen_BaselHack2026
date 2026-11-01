"""Run the live pipeline (align onto the reference, crop the hooks, classify) on one picture.

    .venv/Scripts/python code/scripts/run_model_for_frame.py path/to/picture.jpeg [--camera live01]

Uses the camera's saved layout and reference (data/layouts/<camera>.json and .ref.npz), decides from this one
picture alone, prints the alignment and every hook, and writes <picture>.result.jpg next to it.
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from bikeslot import config, registration  # noqa: E402
from bikeslot.preview import render  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402
from bikeslot.slotcls import pipeline, smoothing  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("image")
ap.add_argument("--camera", default="live01")
args = ap.parse_args()

img = cv2.imread(args.image)
if img is None:
    sys.exit(f"cannot read {args.image}")
layout_path = config.LAYOUT_DIR / f"{args.camera}.json"
ref_path = config.LAYOUT_DIR / f"{args.camera}.ref.npz"
layout = Layout.model_validate_json(layout_path.read_text(encoding="utf-8"))
layout.params.update(k_frames=1, min_frames=1)
ref = registration.Reference.load(ref_path) if ref_path.exists() else None

res = pipeline.analyze_frame(layout, img, args.camera, smoothing.make_trackers(layout), reference=ref)
print("alignment:", res.status.registration if ref else "no reference saved - picture used as it is")
if res.status.message:
    print("message:", res.status.message)
for s in res.status.slots:
    print(f"hook {s.index:2d}: {s.status:8s} p={s.score:.2f}")
out = Path(args.image).with_suffix(".result.jpg")
out.write_bytes(render(res.frame, res.status, res.detections))
print("overlay:", out)
