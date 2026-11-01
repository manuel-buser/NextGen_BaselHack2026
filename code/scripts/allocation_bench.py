"""Classifier alone vs classifier + hook allocation (slotcls/allocate.py) on the labelled live01 pictures and the
Memox photos. Decisions from single pictures (no smoothing). Overlays go to assets/pictures/validationPics/allocation.

    .venv/Scripts/python code/scripts/allocation_bench.py [--tonight <folder>]

live01: the held-out test split of hopefullyUsableDataset (exact hooks) and the six camera frames of 2026-11-01 night
in assets/pictures/nightTestFrames (mirrored by the camera: flipped, then aligned). Memox: their own layouts, bikes
counted by eye.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config, registration  # noqa: E402
from bikeslot.preview import render  # noqa: E402
from bikeslot.schemas import Layout  # noqa: E402
from bikeslot.slotcls import pipeline, smoothing  # noqa: E402

MEMOX_BY_EYE = {"memox_01": 4, "memox_02": 5, "memox_03": 3, "memox_04": 1, "memox_05": 2, "memox_06": 6,
                "memox_07": 7, "memox_08": 5, "memox_09": 6, "memox_10": 2}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tonight", default=str(config.ROOT / "assets" / "pictures" / "nightTestFrames"),
                help="folder with the night frames live01_v2.jpeg, a.jpeg, ... (skipped when missing)")
args = ap.parse_args()
pics = config.ROOT / "assets" / "pictures"
out = pics / "validationPics" / "allocation"
out.mkdir(parents=True, exist_ok=True)

d = pics / "hopefullyUsableDataset" / "aligned"
live = Layout.model_validate_json((d / "live01.layout.json").read_text(encoding="utf-8"))
items = []                                      # (name, picture, layout, true occupied hooks or None, count by eye)
for fr in json.load(open(d / "test.labels.json"))["frames"]:
    items.append(("test_" + fr["file"][16:22], cv2.imread(str(d / fr["file"])), live,
                  {int(k[-2:]) for k, v in fr["slots"].items() if v == "occupied"}, None))
if args.tonight and Path(args.tonight).exists():
    ref = registration.make_reference(cv2.imread(str(pics / "reference" / "live01_reference.jpg")))
    for name, hooks in (("live01_v2", {18}), ("a", {18}), ("live01_rack", {18}), ("live01_v4_now", {18}),
                        ("live01_v3", {18}), ("live01_v6", {10, 18})):
        img = cv2.flip(cv2.imread(str(Path(args.tonight) / f"{name}.jpeg")), 1)
        reg = registration.register(ref, img)
        if reg.ok:
            items.append(("tonight_" + name, registration.warp(img, reg, ref)[0], live, hooks, None))
for lf in sorted(glob.glob(str(pics / "DemoPictures" / "memox_*.labels.json"))):
    lab = json.load(open(lf))
    lay = Layout.model_validate_json((Path(lf).parent / lab["layout"]).read_text(encoding="utf-8"))
    name = Path(lf).stem[:8]
    items.append((name, cv2.imread(str(Path(lf).parent / lab["frames"][0]["file"])), lay, None, MEMOX_BY_EYE[name]))


def run(img, lay, allocation):
    lay = lay.model_copy(deep=True)
    lay.params = {**(lay.params or {}), "k_frames": 1, "min_frames": 1}
    if allocation:
        lay.params["allocation"] = "dino"
    t = time.time()
    res = pipeline.analyze_frame(lay, img, "bench", smoothing.make_trackers(lay), blur_persons=False)
    return res, {s.index for s in res.status.slots if s.status == "occupied"}, (time.time() - t) * 1000


def shown_taken(res):   # what the website shows as taken: occupied and unknown
    return {s.index for s in res.status.slots if s.status != "free"}


score = {a: dict(found=0, bikes=0, wrong=0, free=0, exact=0, err=0, ms=[], bike_free=0, shown_wrong=0, few=0) for a in (False, True)}
lines, memox_tiles = [], []
for name, img, lay, truth, eye in items:
    row = {}
    for alloc in (False, True):
        res, occ, ms = run(img, lay, alloc)
        s = score[alloc]; s["ms"].append(ms)
        if truth is not None:
            s["bikes"] += len(truth); s["found"] += len(truth & occ)
            s["free"] += len(lay.slots) - len(truth); s["wrong"] += len(occ - truth)
            s["bike_free"] += len(truth - shown_taken(res)); s["shown_wrong"] += len(shown_taken(res) - truth)
        else:
            s["exact"] += len(occ) == eye; s["err"] += abs(len(occ) - eye)
            s["few"] += max(0, eye - len(shown_taken(res)))
        row[alloc] = (occ, res)
    occ0, occ1 = row[False][0], row[True][0]
    target = sorted(truth) if truth is not None else f"{eye} by eye"
    lines.append(f"  {name:22s} truth {target}  alone {sorted(occ0)}  ->  allocated {sorted(occ1)}"
                 f"   (vehicles {row[True][1].status.allocation and row[True][1].status.allocation['vehicles']})")
    res = row[True][1]
    pic = cv2.imdecode(np.frombuffer(render(res.frame, res.status, res.detections), np.uint8), cv2.IMREAD_COLOR)
    bar = np.zeros((34, pic.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{name}: classifier alone {len(occ0)} -> with allocation {len(occ1)} | truth {target}", (8, 23), 0, 0.55, (255, 255, 255), 1)
    pic = np.vstack([bar, pic])
    cv2.imwrite(str(out / f"{name}.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if truth is None:
        memox_tiles.append(cv2.resize(pic, (800, int(800 * pic.shape[0] / pic.shape[1]))))

print("\n".join(lines))
for alloc in (False, True):
    s = score[alloc]
    print(f"{'with allocation' if alloc else 'classifier alone':17s} | live01: bikes found {s['found']}/{s['bikes']}, "
          f"free hooks wrongly taken {s['wrong']}/{s['free']} | Memox: count exact {s['exact']}/10, total count error {s['err']} "
          f"| {np.median(s['ms']):.0f} ms per picture")
    print(f"{'':17s} | as the website shows it (unknown = taken): live01 real bikes shown free {s['bike_free']}/{s['bikes']}, "
          f"free hooks shown taken {s['shown_wrong']}/{s['free']} | Memox bikes too few {s['few']}")
if memox_tiles:
    h = max(t.shape[0] for t in memox_tiles)
    memox_tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in memox_tiles]
    for i, sname in ((0, "a"), (6, "b")):
        ch = memox_tiles[i:i + 6]; ch += [np.zeros_like(memox_tiles[0])] * (-len(ch) % 2)
        cv2.imwrite(str(out / f"SHEET_memox_{sname}.jpg"), np.vstack([np.hstack(ch[j:j + 2]) for j in range(0, len(ch), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
