"""Temporal smoothing: majority vote over the last N frames + hysteresis.

Reuses ``occupancy.SlotTracker`` (the vote window with hysteresis of the detector pipeline) and only adds the
probability -> vote step: a classifier output within ``unsure_band`` of 0.5 casts no vote at all, so a
person walking through or a bike half-way into the hook neither flips nor refreshes the state.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from ..occupancy import SlotTracker
from ..schemas import Layout


def vote_from_prob(p: Optional[float], unsure_band: float = config.SLOTCLS_UNSURE_BAND) -> Optional[float]:
    if p is None or abs(p - 0.5) < unsure_band:
        return None
    return 1.0 if p >= 0.5 else 0.0


def make_trackers(layout: Layout) -> dict[str, SlotTracker]:
    p = layout.params or {}
    k = int(p.get("k_frames", config.SLOTCLS_K_FRAMES))
    min_frames = int(p.get("min_frames", config.SLOTCLS_MIN_FRAMES))
    return {s.slot_id: SlotTracker(s.slot_id, k=k, min_frames=min_frames) for s in layout.slots}


def update(trackers: dict[str, SlotTracker], probs: dict[str, Optional[float]],
           unsure_band: float = config.SLOTCLS_UNSURE_BAND) -> None:
    for slot_id, tr in trackers.items():
        tr.update(vote_from_prob(probs.get(slot_id), unsure_band))
