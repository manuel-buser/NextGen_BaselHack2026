"""Feeds the demo racks with made-up, slowly changing occupancy, so the pages can be shown with one real camera.

Standard library only: python tools/demo_data.py --app http://localhost:8000
Asks the app which racks are marked demo (start it with BIKESLOT_RACKS=racks.demo.yaml) and plays a compressed
day in a loop: bikes arrive in the morning, the first demo rack fills up, bikes leave in the evening; with
--outage the second demo rack also loses its camera from 11:00 to 18:00. The demo rack furthest from the main
(first real) rack gets no data, so the page also shows an inactive rack. Real racks are left to their cameras,
unless --feed-real asks to feed them too for a rehearsal without the camera. Each "picture" is a few bytes
spelling out the spaces (DEMO:1101..., 1 = bike), read by the app's built-in stand-in detector. Each picture
carries the simulated date and time (X-Demo-Clock), starting on Monday 2026-11-02, so the app learns a usual
weekday and weekend (quieter); --warmup DAYS first plays that many days before it at full speed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from fake_camera import post

# One profile per demo rack, in the order the app lists them (repeating if there are more racks).
# peak: share of spaces taken at the busiest time; arrive, leave: hours when most bikes come and go;
# offline: simulated hours during which the camera sends nothing (only with --outage).
PROFILES = [
    {"peak": 1.0, "arrive": 7.5, "leave": 17.0},
    {"peak": 0.7, "arrive": 8.0, "leave": 18.0, "offline": (11.0, 18.0)},
    {"peak": 0.55, "arrive": 8.5, "leave": 16.5},
    {"peak": 0.85, "arrive": 7.0, "leave": 17.5},
]
REAL_PROFILE = {"peak": 0.75, "arrive": 7.5, "leave": 17.0}  # busy but rarely full
NIGHT = 0.05  # share of spaces still taken at night
MOVE_CHANCE = 0.35  # per picture, for each bike that still has to arrive or leave
START_DATE = datetime(2026, 11, 2)  # a Monday: the live loop starts here, the warmup days come before it
WEEKEND = 0.3  # Saturday and Sunday: the profile peak times this
WARMUP_STEP = timedelta(minutes=5)  # one picture per rack per this much simulated time


def expected_share(hour: float, profile: dict, weekend: bool = False) -> float:
    arriving = 1 / (1 + math.exp(-(hour - profile["arrive"]) / 0.6))
    leaving = 1 / (1 + math.exp(-(hour - profile["leave"]) / 0.8))
    peak = profile["peak"] * (WEEKEND if weekend else 1)
    return NIGHT + (peak - NIGHT) * arriving * (1 - leaving)


def clock(hour: float) -> str:
    return f"{int(hour) % 24:02d}:{int(hour % 1 * 60):02d}"


def clock_hour(text: str) -> float:
    hours, minutes = text.split(":")
    return int(hours) + int(minutes) / 60


class DemoRack:
    def __init__(self, rack: dict, profile: dict, hour: float, outage: bool = False) -> None:
        self.id = rack["id"]
        self.capacity = rack["capacity"]
        self.profile = profile
        self.outage = outage
        self.taken: set[int] = set()
        self.temp_c = 38.0
        self.step(hour, chance=1.0)  # start at the level expected for that hour, not empty

    def offline(self, hour: float) -> bool:
        window = self.profile.get("offline")
        return self.outage and bool(window) and window[0] <= hour < window[1]

    def step(self, hour: float, chance: float = MOVE_CHANCE, weekend: bool = False) -> None:
        target = round(self.capacity * expected_share(hour, self.profile, weekend))
        free = sorted(set(range(1, self.capacity + 1)) - self.taken)
        for _ in range(target - len(self.taken)):
            if free and random.random() < chance:  # arrivals prefer the spaces near the start of the rack
                slot = random.choices(free, weights=[1 / (k + 1) for k in range(len(free))])[0]
                free.remove(slot)
                self.taken.add(slot)
        for _ in range(len(self.taken) - target):
            if random.random() < chance:
                self.taken.remove(random.choice(sorted(self.taken)))
        self.temp_c = round(min(55.0, max(32.0, self.temp_c + random.uniform(-0.4, 0.5))), 1)

    def picture(self) -> bytes:
        return b"DEMO:" + "".join("1" if slot in self.taken else "0" for slot in range(1, self.capacity + 1)).encode()


class Poster:
    """Sends pictures with the simulated clock over one kept-open connection (the warmup sends thousands)."""

    def __init__(self, app: str) -> None:
        self.app = app
        self.url = urllib.parse.urlsplit(app)
        self.connection: http.client.HTTPConnection | None = None

    def picture(self, rack: DemoRack, moment: datetime) -> int:
        headers = {"Content-Type": "image/jpeg", "X-Node-Id": f"demo-{rack.id}", "X-Demo-Clock": moment.strftime("%Y-%m-%dT%H:%M")}
        for _ in range(2):  # the app closes a connection left idle for a few seconds: open a new one once
            if self.connection is None:
                kind = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
                self.connection = kind(self.url.hostname, self.url.port, timeout=20)
            try:
                self.connection.request("POST", f"{self.url.path.rstrip('/')}/api/camera/{rack.id}/picture", rack.picture(), headers)
                response = self.connection.getresponse()
                response.read()
                return response.status
            except (OSError, http.client.HTTPException) as exc:
                self.connection.close()
                self.connection, error = None, exc
        print(f"cannot reach {self.app}: {error}", file=sys.stderr)
        return 0


def warm_up(racks: list[DemoRack], poster: Poster, start: float, days: int) -> None:
    """Plays the `days` days before the live start without sleeping, so the app has learned days at the pitch."""
    began, first = time.monotonic(), START_DATE + timedelta(hours=start) - timedelta(days=days)
    for day in range(days):
        moment = first + timedelta(days=day)
        while moment < first + timedelta(days=day + 1):
            hour = moment.hour + moment.minute / 60
            for rack in racks:
                rack.step(hour, weekend=moment.weekday() >= 5)
                if not rack.offline(hour) and poster.picture(rack, moment) == 0:
                    sys.exit("warmup stopped: the app cannot be reached")
            moment += WARMUP_STEP
        print(f"warmup day {day + 1}/{days} ({first + timedelta(days=day):%a %Y-%m-%d}) done after {time.monotonic() - began:.0f} s", flush=True)


def get_json(app: str, path: str) -> dict:
    with urllib.request.urlopen(f"{app}{path}", timeout=10) as response:
        return json.load(response)


def distance_m(a: dict, b: dict) -> float:
    dy = (a["lat"] - b["lat"]) * 111_000
    dx = (a["lng"] - b["lng"]) * 111_000 * math.cos(math.radians(a["lat"]))
    return math.hypot(dx, dy)


def furthest(demo: list[dict], origin: dict | None) -> str | None:
    placed = [rack for rack in demo if rack.get("location")]
    if not origin or not origin.get("location") or not placed:
        return None
    return max(placed, key=lambda rack: distance_m(rack["location"], origin["location"]))["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=5.0, help="real seconds between pictures")
    parser.add_argument("--day", type=float, default=360.0, help="real seconds for one simulated day")
    parser.add_argument("--start", type=clock_hour, default="06:30", help="simulated time of day at launch, HH:MM")
    parser.add_argument("--outage", action="store_true", help="the second demo rack loses its camera from 11:00 to 18:00")
    parser.add_argument("--inactive", default="auto", help="demo rack left without data: auto (the one furthest from the main rack), a rack id, or none")
    parser.add_argument("--feed-real", action="store_true", help="also feed the racks not marked demo, for a rehearsal without the camera (not labelled as demo on the page; refused while the app uses a real detector)")
    parser.add_argument("--warmup", type=int, default=0, metavar="DAYS", help="first play this many days before the start at full speed, so the predictions have learned days")
    args = parser.parse_args()

    try:
        everything = get_json(args.app, "/api/racks")["racks"]
        stand_in = get_json(args.app, "/api/admin/status")["detector"]["mode"] == "fake"
    except (urllib.error.URLError, ValueError, KeyError) as exc:
        sys.exit(f"cannot read {args.app}: {exc}")
    demo = [rack for rack in everything if rack.get("demo")]
    real = [rack for rack in everything if not rack.get("demo")]
    if not demo:
        sys.exit("no demo racks found: start the app with BIKESLOT_RACKS=racks.demo.yaml")
    if args.feed_real and not stand_in:
        sys.exit("--feed-real refused: the app uses a real detector, so the real racks belong to their cameras")
    if args.inactive == "auto":
        inactive = furthest(demo, real[0] if real else None)
    elif args.inactive == "none":
        inactive = None
    elif args.inactive in {rack["id"] for rack in demo}:
        inactive = args.inactive
    else:
        sys.exit(f"--inactive {args.inactive}: not a demo rack")
    racks = [DemoRack(rack, REAL_PROFILE, args.start) for rack in real] if args.feed_real else []
    fed_demo = [rack for rack in demo if rack["id"] != inactive]
    racks += [DemoRack(rack, PROFILES[i % len(PROFILES)], args.start, args.outage) for i, rack in enumerate(fed_demo)]
    left = f", leaving {inactive} without data" if inactive else ""
    print(f"feeding {', '.join(rack.id for rack in racks)}{left}: one day takes {args.day:g} s, it is {START_DATE:%a} {clock(args.start)} now")
    poster = Poster(args.app)
    if args.warmup > 0:
        warm_up(racks, poster, args.start, args.warmup)
    launched = time.monotonic()
    while True:
        elapsed = time.monotonic() - launched
        hour = (args.start + elapsed / args.day * 24) % 24
        moment = START_DATE + timedelta(hours=args.start + elapsed / args.day * 24)
        report = []
        for rack in racks:
            rack.step(hour, weekend=moment.weekday() >= 5)
            if rack.offline(hour):
                report.append(f"{rack.id} camera offline")
                continue
            node = f"demo-{rack.id}"
            alive = json.dumps({"tempC": rack.temp_c, "tooHot": False, "uptimeS": int(elapsed)}).encode()
            post(f"{args.app}/api/camera/{rack.id}/alive", alive, "application/json", node)
            code = poster.picture(rack, moment)
            report.append(f"{rack.id} {len(rack.taken)}/{rack.capacity}" + ("" if code == 200 else f" (HTTP {code})"))
        print(f"{moment:%a} {clock(hour)} simulated | " + " | ".join(report), flush=True)  # visible at once in a log file too
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
