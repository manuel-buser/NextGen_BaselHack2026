"""Fake ESP32: send snapshots to the bikeslot service.

Examples (from the repo root):
    # loop the golden images into camera cam01 every 3 s
    .venv/Scripts/python code/scripts/replay.py --camera cam01 --source assets/pictures/GoldenDataSet --interval 3 --loop

    # send one file once, resized like an ESP32 UXGA still
    .venv/Scripts/python code/scripts/replay.py --camera cam02 --source some.jpg --resize 1600x1200 --once

    # poll a real ESP32-CAM running the stock CameraWebServer and forward its /capture stills
    .venv/Scripts/python code/scripts/replay.py --camera cam01 --source http://192.168.1.50/capture --interval 5 --loop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot.geometry import imread  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_sources(source: str):
    if source.startswith(("http://", "https://")):
        while True:
            yield source
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            sys.exit(f"no images in {p}")
        yield from files
    elif p.is_file():
        yield p
    else:
        sys.exit(f"source not found: {source}")


def load_bytes(src, resize: str | None, quality: int) -> bytes:
    if isinstance(src, str) and src.startswith("http"):
        r = requests.get(src, timeout=10)
        r.raise_for_status()
        data = r.content
        if not resize:
            return data
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = imread(str(src))
        if img is None:
            sys.exit(f"cannot read {src}")
    if resize:
        w, h = (int(v) for v in resize.lower().split("x"))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000",
                    help="use an IP, not 'localhost': on Windows 'localhost' tries IPv6 first and costs ~2 s per new connection")
    ap.add_argument("--camera", default="cam01")
    ap.add_argument("--source", required=True, help="image file, directory of images, or http URL of an ESP32 /capture endpoint")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between snapshots")
    ap.add_argument("--loop", action="store_true", help="repeat forever (directory/file sources)")
    ap.add_argument("--once", action="store_true", help="send each image once, no waiting after the last one")
    ap.add_argument("--resize", default=None, help="WxH, e.g. 1600x1200 to mimic an ESP32 still")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality when re-encoding")
    ap.add_argument("--multipart", action="store_true", help="send as multipart form field 'file' instead of raw body")
    args = ap.parse_args()

    url = f"{args.server.rstrip('/')}/cameras/{args.camera}/frames"
    sess = requests.Session()
    sent = 0
    while True:
        for src in iter_sources(args.source):
            data = load_bytes(src, args.resize, args.quality)
            t = time.time()
            try:
                if args.multipart:
                    r = sess.post(url, files={"file": ("frame.jpg", data, "image/jpeg")}, timeout=60)
                else:
                    r = sess.post(url, data=data, headers={"Content-Type": "image/jpeg"}, timeout=60)
            except requests.RequestException as exc:
                print(f"!! {exc}")
                time.sleep(args.interval)
                continue
            dt = (time.time() - t) * 1000
            sent += 1
            name = src if isinstance(src, str) else src.name
            if r.ok:
                s = r.json()
                states = "".join({"free": ".", "occupied": "#", "unknown": "?"}[x["status"]] for x in s.get("slots", []))
                print(f"{sent:4d} {name:32s} {len(data)//1024:4d} kB  {dt:6.0f} ms  free {s['free']}/{s['slot_count']}"
                      f"  occ {s['occupied']}  unk {s['unknown']}  [{states}]  {s.get('message','')}")
            else:
                print(f"{sent:4d} {name:32s} HTTP {r.status_code}: {r.text[:200]}")
            if not (args.once and not args.loop):
                time.sleep(args.interval)
        if not args.loop:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
