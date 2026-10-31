"""Stand-in for the camera node: posts a picture and an alive message every few seconds.

Standard library only: python tools/fake_camera.py --app http://localhost:8000 --rack A2 --interval 5
With --images DIR it sends real JPEG files in turn. Without it, synthetic bytes whose length steers the
built-in fake detector; the taken count random-walks between 0 and --capacity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, data: bytes, content_type: str, node_id: str) -> int:
    request = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": content_type, "X-Node-Id": node_id}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except urllib.error.URLError as exc:
        print(f"cannot reach {url}: {exc.reason}", file=sys.stderr)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default="http://localhost:8000")
    parser.add_argument("--rack", default="A2")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between pictures")
    parser.add_argument("--images", help="folder with JPEG files to send in turn")
    parser.add_argument("--capacity", type=int, default=15, help="only for synthetic pictures")
    parser.add_argument("--taken", type=int, default=5, help="starting number of taken slots (synthetic)")
    parser.add_argument("--node", default="fake-camera-1")
    args = parser.parse_args()

    files = itertools.cycle(sorted(Path(args.images).glob("*.jp*g"))) if args.images else None
    taken = args.taken
    temp_c = 38.0
    while True:
        if files:
            path = next(files)
            picture = path.read_bytes()
            label = path.name
        else:
            taken = min(args.capacity, max(0, taken + random.choice((-1, 0, 0, 1))))
            picture = b"FAKEJPEG" + b"\0" * (100 * (args.capacity + 1) + taken - 8)
            label = f"synthetic, {taken} taken"
        temp_c = round(min(80.0, max(30.0, temp_c + random.uniform(-0.5, 0.7))), 1)
        alive = json.dumps({"tempC": temp_c, "tooHot": temp_c > 75.0, "uptimeS": int(time.monotonic())}).encode()
        code_alive = post(f"{args.app}/api/camera/{args.rack}/alive", alive, "application/json", args.node)
        code_pic = post(f"{args.app}/api/camera/{args.rack}/picture", picture, "image/jpeg", args.node)
        print(f"{time.strftime('%H:%M:%S')} rack {args.rack}: alive {code_alive}, picture {code_pic} ({label}, {temp_c} C)")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
