"""Compare vehicle detectors on the labelled test pictures of live01: how many vehicles stand in the rack area
(wheels = bottom of the box inside the area) against the true count, and the time per picture.

    .venv/Scripts/python code/scripts/vehicle_bench.py --detectors yolo dino [--tonight <folder with live01_*.jpeg>]

Pictures: the held-out test split of assets/pictures/hopefullyUsableDataset/aligned (one bike each) and, with
--tonight, the six camera frames of 2026-11-01 night (mirrored by the camera: flipped, then aligned).
Rack area: vehicles.rack_area (the slot row widened by one hook at each end, down to 2.2 x its depth).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config, registration, vehicles  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--detectors", nargs="+", default=["yolo", "dino"])
ap.add_argument("--tonight", help="folder with live01_v2.jpeg, live01_v6.jpeg, ... (optional)")
args = ap.parse_args()

pics = config.ROOT / "assets" / "pictures"
d = pics / "hopefullyUsableDataset" / "aligned"
layout = Layout.model_validate_json((d / "live01.layout.json").read_text(encoding="utf-8"))
items = [(f"test {f['file'][16:22]}", cv2.imread(str(d / f["file"])), 1)
         for f in json.load(open(d / "test.labels.json"))["frames"]]
if args.tonight:
    ref = registration.make_reference(cv2.imread(str(pics / "reference" / "live01_reference.jpg")))
    for name, n in (("live01_v2", 1), ("a", 1), ("live01_rack", 1), ("live01_v4_now", 1), ("live01_v3", 1), ("live01_v6", 2)):
        img = cv2.flip(cv2.imread(str(Path(args.tonight) / f"{name}.jpeg")), 1)
        reg = registration.register(ref, img)
        if reg.ok:
            items.append((f"tonight {name}", registration.warp(img, reg, ref)[0], n))

area = vehicles.rack_area(layout, (items[0][1].shape[1], items[0][1].shape[0]))

for name in args.detectors:
    det = vehicles.get(name)
    t0 = time.time(); det.load(); load_s = time.time() - t0
    det.detect(items[0][1])                                   # warm-up
    exact = over = under = 0; times = []; lines = []
    for label, img, truth in items:
        t = time.time(); dets = det.detect(img); times.append((time.time() - t) * 1000)
        inside = [x for x in dets if vehicles.in_area(x, area)]
        n = len(inside)
        exact += n == truth; over += n > truth; under += n < truth
        if n != truth:
            lines.append(f"    {label}: counted {n}, true {truth}  ({', '.join(f'{x.name} {x.conf:.2f}' for x in inside) or '-'})")
    print(f"{name}: load {load_s:.1f} s, {np.median(times):.0f} ms per picture (median) | count right {exact}/{len(items)}, "
          f"too many {over}, too few {under}")
    print("\n".join(lines))
