"""Temporal smoothing: ``occupancy.SlotTracker`` (vote window + hysteresis) fed with classifier probabilities.

A probability within ``unsure_band`` of 0.5 casts no vote, so a passer-by or a half-parked bike neither flips nor
refreshes the state.
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
    """Per camera (layout.params): k_frames, min_frames, and occupied_above / free_below for the vote mean - e.g.
    k_frames 2, occupied_above 0.5, free_below 0.0: occupied on the first occupied vote, free only after two free
    votes in a row (a wrong "free" is the error that must not happen)."""
    p = layout.params or {}
    k = int(p.get("k_frames", config.SLOTCLS_K_FRAMES))
    min_frames = int(p.get("min_frames", config.SLOTCLS_MIN_FRAMES))
    above, below = p.get("occupied_above"), p.get("free_below")
    return {s.slot_id: SlotTracker(s.slot_id, k=k, min_frames=min_frames,
                                   occupied_above=None if above is None else float(above),
                                   free_below=None if below is None else float(below))
            for s in layout.slots}


def update(trackers: dict[str, SlotTracker], probs: dict[str, Optional[float]],
           unsure_band: float = config.SLOTCLS_UNSURE_BAND) -> None:
    for slot_id, tr in trackers.items():
        tr.update(vote_from_prob(probs.get(slot_id), unsure_band))
