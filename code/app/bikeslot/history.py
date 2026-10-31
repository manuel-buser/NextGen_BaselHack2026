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
FADE_MIN = 60  # arrival: today's difference from the usual day halves with every this many minutes ahead


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
        self.groups: dict[str, list[str]] = {}  # location id -> its racks' ids, whose samples are added per minute

    def record(self, rack_id: str, moment: datetime, free: int, capacity: int) -> None:
        self.samples.setdefault(rack_id, {})[moment.strftime(MINUTE)] = (free, capacity)
        self._profiles.pop(rack_id, None)
        for group, parts in self.groups.items():
            if rack_id in parts:
                self._profiles.pop(group, None)
        self.changed = True

    def samples_of(self, key: str) -> dict[str, tuple[int, int]]:
        """A rack's samples, or a location's: per minute the sum over its racks with a sample in that minute (a
        rack without one adds no free spaces)."""
        if key not in self.groups:
            return self.samples.get(key, {})
        summed: dict[str, tuple[int, int]] = {}
        for part in self.groups[key]:
            for minute, (free, capacity) in self.samples.get(part, {}).items():
                had_free, had_capacity = summed.get(minute, (0, 0))
                summed[minute] = (had_free + free, had_capacity + capacity)
        return summed

    def profile(self, rack_id: str, kind: str) -> tuple[list[dict], int]:
        """Per 15-minute slot of a `kind` day: average free, share of samples with the rack full, sample count and
        distinct days; plus the number of distinct days of that kind."""
        cached = self._profiles.setdefault(rack_id, {})
        if kind not in cached:
            sums, fulls, counts = [0] * SLOTS, [0] * SLOTS, [0] * SLOTS
            slot_days: list[set[str]] = [set() for _ in range(SLOTS)]
            kinds: dict[str, str] = {}
            for minute, (free, _capacity) in self.samples_of(rack_id).items():
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
        insights = self.insights(rack.id, datetime.fromtimestamp(now))
        return {key: insights[key] for key in ("usually_full_by", "usual_free_now", "days", "learning")}

    def usual_free(self, rack_id: str, moment: datetime) -> float | None:
        here = self.profile(rack_id, day_type(moment))[0][slot_of(moment)]
        return here["free"] if here["n"] else None

    def arrival(self, rack, now: float, free_now: int | None, minutes: int) -> dict:
        """Free spaces expected `minutes` from now: the usual count then plus today's difference from the usual count
        now, fading with the time ahead. Without a usual count at either time: the count now."""
        moment = datetime.fromtimestamp(now)
        then = moment + timedelta(minutes=minutes)
        free = None
        if free_now is not None:
            usual_now, usual_then = self.usual_free(rack.id, moment), self.usual_free(rack.id, then)
            guess = free_now
            if usual_now is not None and usual_then is not None:
                guess = usual_then + (free_now - usual_now) * 0.5 ** (minutes / FADE_MIN)
            free = round(min(rack.capacity, max(0, guess)))
        return {"minutes": minutes, "at": then.strftime("%H:%M"), "free": free}

    def view(self, rack, now: float, free_now: int | None) -> dict:
        moment = datetime.fromtimestamp(now)
        kind = day_type(moment)
        slots, _days = self.profile(rack.id, kind)
        today, until = moment.strftime("%Y-%m-%d"), moment.strftime(MINUTE)
        sofar: dict[int, list[int]] = {}
        for minute, (free, _capacity) in self.samples_of(rack.id).items():
            if minute[:10] == today and minute <= until:
                sofar.setdefault((int(minute[11:13]) * 60 + int(minute[14:16])) // SLOT_MIN, []).append(free)
        return {
            "rack": rack.id,
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
