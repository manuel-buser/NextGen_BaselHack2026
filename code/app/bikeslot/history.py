"""Free counts per rack over time and what they say about a usual day. Only the count per rack is kept, never
which space was used when."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

SLOT_MIN = 15
SLOTS = 24 * 60 // SLOT_MIN
KEEP = timedelta(weeks=8)
MINUTE = "%Y-%m-%dT%H:%M"  # one sample per rack per minute, the latest in that minute wins
FULL_SHARE = 0.5  # a slot counts as usually full when the rack was full in at least this share of samples
MIN_DAYS = 2  # ... and those samples come from at least this many days
BEST_FREE = 2  # best_before: the last slot before that with at least this many free on average


def parse_clock(text: str | None) -> datetime | None:
    """The demo driver's simulated local time, for example 2026-11-02T07:35; None when missing or malformed."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.strip()).replace(tzinfo=None, second=0, microsecond=0)
    except ValueError:
        return None


def day_type(moment: datetime) -> str:
    return "weekend" if moment.weekday() >= 5 else "weekday"


def slot_of(moment: datetime) -> int:
    return (moment.hour * 60 + moment.minute) // SLOT_MIN


def slot_label(slot: int) -> str:
    return f"{slot * SLOT_MIN // 60:02d}:{slot * SLOT_MIN % 60:02d}"


class History:
    def __init__(self, path: str | None = None) -> None:
        self.path = path  # None: kept in memory only
        self.samples: dict[str, dict[str, tuple[int, int]]] = {}  # rack id -> minute -> (free, capacity)
        self.changed = False
        self._profiles: dict[str, dict[str, tuple[list[dict], int]]] = {}  # rack id -> day type -> profile

    @staticmethod
    def moment(rack, now: float) -> datetime:
        """'Now' for a rack: the simulated time of a demo rack fed with a clock, else the real local time."""
        return rack.demo_clock or datetime.fromtimestamp(now)

    def record(self, rack_id: str, moment: datetime, free: int, capacity: int) -> None:
        self.samples.setdefault(rack_id, {})[moment.strftime(MINUTE)] = (free, capacity)
        self._profiles.pop(rack_id, None)
        self.changed = True

    def profile(self, rack_id: str, kind: str) -> tuple[list[dict], int]:
        """Per 15-minute slot of a `kind` day: average free, share of samples with the rack full, sample count and
        distinct days; plus the number of distinct days of that kind."""
        cached = self._profiles.setdefault(rack_id, {})
        if kind not in cached:
            sums, fulls, counts = [0] * SLOTS, [0] * SLOTS, [0] * SLOTS
            slot_days: list[set[str]] = [set() for _ in range(SLOTS)]
            kinds: dict[str, str] = {}
            for minute, (free, _capacity) in self.samples.get(rack_id, {}).items():
                date = minute[:10]
                if date not in kinds:
                    kinds[date] = day_type(datetime.strptime(date, "%Y-%m-%d"))
                if kinds[date] != kind:
                    continue
                slot = (int(minute[11:13]) * 60 + int(minute[14:16])) // SLOT_MIN
                sums[slot] += free
                fulls[slot] += free == 0
                counts[slot] += 1
                slot_days[slot].add(date)
            slots = [
                {
                    "free": sums[i] / counts[i] if counts[i] else None,
                    "full_share": fulls[i] / counts[i] if counts[i] else 0.0,
                    "n": counts[i],
                    "days": len(slot_days[i]),
                }
                for i in range(SLOTS)
            ]
            cached[kind] = (slots, sum(1 for k in kinds.values() if k == kind))
        return cached[kind]

    def insights(self, rack_id: str, moment: datetime) -> dict:
        slots, days = self.profile(rack_id, day_type(moment))
        full_by = next((i for i, s in enumerate(slots) if s["days"] >= MIN_DAYS and s["full_share"] >= FULL_SHARE), None)
        best = None
        if full_by is not None:
            best = next((i for i in range(full_by - 1, -1, -1) if slots[i]["n"] and slots[i]["free"] >= BEST_FREE), None)
        here = slots[slot_of(moment)]
        return {
            "usually_full_by": None if full_by is None else slot_label(full_by),
            "best_before": None if best is None else slot_label(best),
            "usual_free_now": round(here["free"], 1) if here["n"] else None,
            "days": days,
            "learning": days < MIN_DAYS,
        }

    def forecast(self, rack, now: float) -> dict:
        insights = self.insights(rack.id, self.moment(rack, now))
        return {key: insights[key] for key in ("usually_full_by", "usual_free_now", "days", "learning")}

    def view(self, rack, now: float, free_now: int | None) -> dict:
        moment = self.moment(rack, now)
        kind = day_type(moment)
        slots, _days = self.profile(rack.id, kind)
        today, until = moment.strftime("%Y-%m-%d"), moment.strftime(MINUTE)
        sofar: dict[int, list[int]] = {}
        for minute, (free, _capacity) in self.samples.get(rack.id, {}).items():
            if minute[:10] == today and minute <= until:
                sofar.setdefault((int(minute[11:13]) * 60 + int(minute[14:16])) // SLOT_MIN, []).append(free)
        return {
            "rack": rack.id,
            "clock": "demo" if rack.demo_clock else "real",
            "now": moment.strftime("%H:%M"),
            "day_type": kind,
            "free_now": free_now,
            "capacity": rack.capacity,
            "today": [{"t": slot_label(slot), "free": round(sum(frees) / len(frees), 1)} for slot, frees in sorted(sofar.items())],
            "typical": [
                {"t": slot_label(i), "free": None if s["free"] is None else round(s["free"], 1), "full_share": round(s["full_share"], 2), "n": s["n"]}
                for i, s in enumerate(slots)
            ],
            "insights": self.insights(rack.id, moment),
        }

    def prune(self, now: datetime) -> None:
        cutoff = (now - KEEP).strftime(MINUTE)  # this text format sorts like the time it names
        for rack_id, minutes in self.samples.items():
            self.samples[rack_id] = {minute: sample for minute, sample in minutes.items() if minute >= cutoff}
        self._profiles.clear()

    def load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as handle:
                raw = json.load(handle)["racks"]
            samples = {}
            for rack_id, minutes in raw.items():
                for minute in minutes:
                    datetime.strptime(minute, MINUTE)  # a malformed time would break every later request
                samples[str(rack_id)] = {minute: (int(free), int(capacity)) for minute, (free, capacity) in minutes.items()}
        except FileNotFoundError:
            log.warning("history: no file at %s yet, starting empty", self.path)
            return
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("history: cannot read %s (%s), starting empty", self.path, exc)
            return
        self.samples = samples
        self.prune(datetime.now())

    def save(self) -> None:
        """Write atomically (temporary file, then rename) so a crash mid-write never leaves half a file."""
        if not self.path or not self.changed:
            return
        self.prune(datetime.now())
        temporary = f"{self.path}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({"racks": {rack_id: dict(sorted(m.items())) for rack_id, m in self.samples.items()}}, handle)
            os.replace(temporary, self.path)
        except OSError as exc:
            log.warning("history: cannot write %s (%s), trying again later", self.path, exc)
            return
        self.changed = False
