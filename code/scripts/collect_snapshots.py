"""Save every new picture the relay holds into assets/pictures/rawTrainingData (training data for the classifier).

    .venv/Scripts/python code/scripts/collect_snapshots.py            # run until Ctrl+C
    .venv/Scripts/python code/scripts/collect_snapshots.py --once     # save the current picture and stop

Reads the snapshot URL (first entry of BIKESLOT_PULL_CAMERAS) and BIKESLOT_PULL_VIEW_KEY from .env. Asks with
If-None-Match, so a picture is saved once however often it is polled; pictures older than --max-age are skipped
(the camera stopped sending). Files are named <camera>_<UTC time of saving>.jpg.
These are raw pictures: people and plates are NOT blurred.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config, pull  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(config.ROOT / "assets" / "pictures" / "rawTrainingData"))
ap.add_argument("--interval", type=float, default=0.5, help="seconds between polls")
ap.add_argument("--max-age", type=float, default=10.0, help="skip pictures older than this many seconds")
ap.add_argument("--once", action="store_true")
args = ap.parse_args()

camera, url = next(iter(pull.parse(config.PULL_CAMERAS).items()))
headers = {"X-View-Key": config.PULL_VIEW_KEY} if config.PULL_VIEW_KEY else {}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(f"saving new pictures of {camera} from {pull.redact(url)} into {out}")

etag, saved, sess = None, 0, requests.Session()
while True:
    try:
        r = sess.get(url, headers={**headers, **({"If-None-Match": etag} if etag else {})}, timeout=10)
        age = float(r.headers.get("X-Snapshot-Age-S", 0) or 0)
        if r.status_code == 200 and age <= args.max_age:
            etag = r.headers.get("ETag")
            name = f"{camera}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"[:-3] + ".jpg"
            (out / name).write_bytes(r.content)
            saved += 1
            print(f"{saved:6d}  {name}  {len(r.content) // 1024} KB  age {age:.1f} s", flush=True)
        elif r.status_code not in (200, 304):
            print(f"relay answered {r.status_code}", flush=True)
        elif r.status_code == 200:
            print(f"newest picture is {age:.0f} s old - skipped (camera not sending?)", flush=True)
    except requests.RequestException as exc:
        print(f"relay unreachable: {pull.redact(str(exc))}", flush=True)
    if args.once:
        break
    time.sleep(args.interval)
