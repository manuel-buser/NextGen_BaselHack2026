"""Feeds the demo racks with made-up, slowly changing occupancy, so the pages can be shown with one real camera.

Standard library only: python tools/demo_data.py --app http://localhost:8000
Asks the app which racks are marked demo (start it with BIKESLOT_RACKS=racks.demo.yaml) and feeds them at the
real time of day: bikes arrive in the morning, the section nearest the venue entrance fills up, bikes leave in
the evening, and a bike or two comes or goes every minute; Saturdays and Sundays are quieter. With --outage the
second demo rack also loses its camera from 11:00 to 18:00. The demo rack furthest from the real camera gets no
data, so the page also shows an inactive rack. Real racks are left to their cameras, unless --feed-real asks to
feed them too for a rehearsal without the camera. Each "picture" is a few bytes spelling out the spaces
(DEMO:1101..., 1 = bike), read by the app's built-in stand-in detector.

--seed [DAYS] instead writes DAYS (default 7) days of history up to now into the history file, for every rack
that gets data and for the real racks too, then exits. Run it before the app starts: a running app keeps its own
copy of the history and would overwrite the file. It reads the racks file with the app's code (uv run).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from fake_camera import post

# One profile per fed demo rack, in the order the app lists them (repeating if there are more racks). With
# racks.demo.yaml: Peter Merian-Weg sections 1, 3 and 4 from west to east (the west end, nearest the venue
# entrance, fills first and fullest, the east end stays emptier), then B.
# peak: share of spaces taken at the busiest time; arrive, leave: hours when most bikes come and go;
# offline: hours during which the camera sends nothing (only with --outage).
PROFILES = [
    {"peak": 1.0, "arrive": 7.5, "leave": 17.0},
    {"peak": 0.6, "arrive": 8.0, "leave": 17.0, "offline": (11.0, 18.0)},
    {"peak": 0.35, "arrive": 8.5, "leave": 16.5},
    {"peak": 0.7, "arrive": 8.0, "leave": 18.0},
]
REAL_PROFILE = {"peak": 0.85, "arrive": 7.75, "leave": 17.5}  # section 2: busy but rarely full
NIGHT = 0.05  # share of spaces still taken at night
MOVE_CHANCE = 0.35  # per picture, for each bike that still has to arrive or leave
CHURN = 1  # every minute the level moves by up to this many bikes either way, so the counts never stand still
WEEKEND = 0.3  # Saturday and Sunday: the profile peak times this


def hour_of(moment: datetime) -> float:
    return moment.hour + moment.minute / 60


def expected_share(hour: float, profile: dict, weekend: bool = False) -> float:
    arriving = 1 / (1 + math.exp(-(hour - profile["arrive"]) / 0.6))
    leaving = 1 / (1 + math.exp(-(hour - profile["leave"]) / 0.8))
    peak = profile["peak"] * (WEEKEND if weekend else 1)
    return NIGHT + (peak - NIGHT) * arriving * (1 - leaving)


class DemoRack:
    def __init__(self, rack: dict, profile: dict, moment: datetime, outage: bool = False) -> None:
        self.id = rack["id"]
        self.capacity = rack["capacity"]
        self.profile = profile
        self.outage = outage
        self.taken: set[int] = set()
        self.temp_c = 38.0
        self.minute: datetime | None = None
        self.wobble = 0
        self.step(moment, chance=1.0)  # start at the level expected for that time, not empty

    def offline(self, moment: datetime) -> bool:
        window = self.profile.get("offline")
        return self.outage and bool(window) and window[0] <= hour_of(moment) < window[1]

    def step(self, moment: datetime, chance: float = MOVE_CHANCE) -> None:
        minute = moment.replace(second=0, microsecond=0)
        if minute != self.minute:
            self.minute, self.wobble = minute, random.randint(-CHURN, CHURN)
        share = expected_share(hour_of(moment), self.profile, moment.weekday() >= 5)
        target = min(self.capacity, max(0, round(self.capacity * share) + self.wobble))
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


def get_json(app: str, path: str) -> dict:
    with urllib.request.urlopen(f"{app}{path}", timeout=10) as response:
        return json.load(response)


def answering(app: str) -> bool:
    try:
        with urllib.request.urlopen(f"{app}/api/health", timeout=3):
            return True
    except urllib.error.HTTPError:
        return True  # something answers there, even if it is not this app
    except OSError:  # also URLError: nothing listens there
        return False


def distance_m(a: dict, b: dict) -> float:
    dy = (a["lat"] - b["lat"]) * 111_000
    dx = (a["lng"] - b["lng"]) * 111_000 * math.cos(math.radians(a["lat"]))
    return math.hypot(dx, dy)


def furthest(demo: list[dict], origin: dict | None) -> str | None:
    placed = [rack for rack in demo if rack.get("location")]
    if not origin or not origin.get("location") or not placed:
        return None
    return max(placed, key=lambda rack: distance_m(rack["location"], origin["location"]))["id"]


def split(everything: list[dict], choice: str) -> tuple[list[dict], list[dict], str | None]:
    """The real racks, the demo racks that get data, and the demo rack left without data (`--inactive`)."""
    demo = [rack for rack in everything if rack.get("demo")]
    real = [rack for rack in everything if not rack.get("demo")]
    if not demo:
        sys.exit("no demo racks found: use racks.demo.yaml (BIKESLOT_RACKS for the app, --racks for --seed)")
    if choice == "auto":
        inactive = furthest(demo, real[0] if real else None)
    elif choice == "none":
        inactive = None
    elif choice in {rack["id"] for rack in demo}:
        inactive = choice
    else:
        sys.exit(f"--inactive {choice}: not a demo rack")
    return real, [rack for rack in demo if rack["id"] != inactive], inactive


def demo_racks(real: list[dict], fed_demo: list[dict], moment: datetime, outage: bool = False) -> list[DemoRack]:
    racks = [DemoRack(rack, REAL_PROFILE, moment) for rack in real]
    return racks + [DemoRack(rack, PROFILES[i % len(PROFILES)], moment, outage) for i, rack in enumerate(fed_demo)]


def seed(args: argparse.Namespace) -> None:
    """Write the history the app would have learned from `args.seed` days of the live feed, in its own file format."""
    if args.seed < 1:
        sys.exit("--seed needs at least 1 day")
    if answering(args.app):
        sys.exit(f"--seed refused: an app answers at {args.app}. Stop it first, it would overwrite the seeded file.")
    if Path(args.history).exists():
        sys.exit(f"--seed refused: {args.history} exists. Delete it first, or pass another --history.")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # imported here so the live feed stays standard library only
    from bikeslot.config import Settings
    from bikeslot.history import History
    from bikeslot.state import Store

    store = Store.from_yaml(args.racks, Settings(racks_file=args.racks))
    everything = [{"id": r.id, "capacity": r.capacity, "demo": r.demo, "location": r.location} for r in store.racks.values()]
    real, fed_demo, inactive = split(everything, args.inactive)
    now = datetime.now().replace(second=0, microsecond=0)
    first = now - timedelta(days=args.seed)
    history = History(args.history)
    for rack in demo_racks(real, fed_demo, first):
        moment = first
        while moment <= now:
            rack.step(moment, chance=1.0)  # the live feed's dozen pictures a minute nearly always reach the level
            history.record(rack.id, moment, rack.capacity - len(rack.taken), rack.capacity)
            moment += timedelta(minutes=1)
    history.save()
    if history.changed:
        sys.exit(f"cannot write {args.history}")
    left = f", {inactive} left without data" if inactive else ""
    print(f"wrote {args.seed} days up to {now:%a %H:%M} for {', '.join(history.samples)}{left} into {args.history}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between pictures")
    parser.add_argument("--outage", action="store_true", help="the second demo rack loses its camera from 11:00 to 18:00")
    parser.add_argument("--inactive", default="auto", help="demo rack left without data: auto (the one furthest from the real camera), a rack id, or none")
    parser.add_argument("--feed-real", action="store_true", help="also feed the racks not marked demo, for a rehearsal without the camera (not labelled as demo on the page; refused while the app uses a real detector)")
    parser.add_argument("--seed", type=int, nargs="?", const=7, metavar="DAYS", help="instead of feeding, write DAYS (default 7) days of history up to now for the fed and the real racks, then exit; only while no app runs")
    parser.add_argument("--racks", default="racks.demo.yaml", help="racks file read by --seed")
    parser.add_argument("--history", default="data/history.json", help="history file written by --seed, the app's HISTORY_FILE")
    args = parser.parse_args()
    if args.seed is not None:
        seed(args)
        return

    try:
        status = get_json(args.app, "/api/admin/status")
        everything = status["racks"]  # one entry per rack, also for the racks the user page shows as one location
        stand_in = status["detector"]["mode"] == "fake"
    except (urllib.error.URLError, ValueError, KeyError) as exc:
        sys.exit(f"cannot read {args.app}: {exc}")
    real, fed_demo, inactive = split(everything, args.inactive)
    if args.feed_real and not stand_in:
        sys.exit("--feed-real refused: the app uses a real detector, so the real racks belong to their cameras")
    racks = demo_racks(real if args.feed_real else [], fed_demo, datetime.now(), args.outage)
    left = f", leaving {inactive} without data" if inactive else ""
    print(f"feeding {', '.join(rack.id for rack in racks)}{left} at the real time of day")
    launched = time.monotonic()
    while True:
        moment = datetime.now()
        report = []
        for rack in racks:
            rack.step(moment)
            if rack.offline(moment):
                report.append(f"{rack.id} camera offline")
                continue
            node = f"demo-{rack.id}"
            alive = json.dumps({"tempC": rack.temp_c, "tooHot": False, "uptimeS": int(time.monotonic() - launched)}).encode()
            post(f"{args.app}/api/camera/{rack.id}/alive", alive, "application/json", node)
            code = post(f"{args.app}/api/camera/{rack.id}/picture", rack.picture(), "image/jpeg", node)
            report.append(f"{rack.id} {len(rack.taken)}/{rack.capacity}" + ("" if code == 200 else f" (HTTP {code})"))
        print(f"{moment:%a %H:%M:%S} | " + " | ".join(report), flush=True)  # visible at once in a log file too
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
