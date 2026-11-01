"""Free the hooks that can only be spill-over: the classifier's echo of a bike body on the neighbouring hooks.

Hooks scoring at least ALLOC_FREE_BELOW form clusters. One bike lights its own hook and at most ALLOC_BIKE_REACH on
each side (measured). With v vehicles over a cluster, a hook is freed only when no placement of v bikes lighting the
whole cluster puts a bike on it. Everything else stays with the classifier; this step only frees hooks.

    decide(probs, vehicle_spans)   # probs: {hook index: P(occupied)}, spans: (left, right) of each box in hook indices
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import config

TAKEN, FREE = "taken", "free"


@dataclass
class Allocation:
    decisions: dict[int, str]                       # hook index -> free (missing: the classifier decides)
    clusters: list[dict] = field(default_factory=list)   # {"hooks", "vehicles", "possible": hooks a bike can be on}

    def info(self) -> dict:
        return {"clusters": self.clusters, "freed": sorted(i for i, d in self.decisions.items() if d == FREE)}


def clusters_of(probs: dict[int, float], free_below: float) -> list[list[int]]:
    out, run = [], []
    for i in sorted(probs):
        if probs[i] >= free_below and (not run or i == run[-1] + 1):
            run.append(i)
        else:
            if run:
                out.append(run)
            run = [i] if probs[i] >= free_below else []
    if run:
        out.append(run)
    return out


def possible_bike_hooks(cluster: list[int], vehicles: int, reach: int) -> list[int]:
    """The hooks of ``cluster`` (consecutive) that some placement of ``vehicles`` bikes, each lighting its own hook and
    ``reach`` hooks on each side, puts a bike on while lighting the whole cluster. Empty: no placement lights it all.

    A placement with a bike on hook j needs that bike plus the fewest bikes lighting the rest left and right of it;
    extra bikes fit anywhere, so j is possible as soon as that minimum is at most ``vehicles``."""
    n, width = len(cluster), 2 * reach + 1
    if vehicles >= n:
        return list(cluster)
    out = []
    for j, hook in enumerate(cluster):
        left, right = max(0, j - reach), max(0, n - 1 - j - reach)
        if -(-left // width) + 1 + -(-right // width) <= vehicles:
            out.append(hook)
    return out


def decide(probs: dict[int, float], vehicle_spans: list[tuple[float, float]], free_below: Optional[float] = None,
           reach: Optional[int] = None) -> Allocation:
    """``vehicle_spans``: each vehicle's box from its left to its right edge, in (fractional) hook indices."""
    free_below = config.ALLOC_FREE_BELOW if free_below is None else free_below
    reach = config.ALLOC_BIKE_REACH if reach is None else reach
    clusters = clusters_of(probs, free_below)
    count = [0] * len(clusters)
    for lo, hi in vehicle_spans:
        x = (lo + hi) / 2
        near = [(min(abs(x - i) for i in c), k) for k, c in enumerate(clusters) if c[0] - 1 <= x <= c[-1] + 1]
        if near:
            count[min(near)[1]] += 1
    alloc = Allocation({})
    for cluster, vehicles in zip(clusters, count):
        possible = possible_bike_hooks(cluster, vehicles, reach) if vehicles else []
        if not possible:   # no vehicle over it, or more bikes than vehicles: nothing is known to be spill-over
            possible = list(cluster)
        alloc.clusters.append({"hooks": cluster, "vehicles": vehicles, "possible": possible})
        alloc.decisions.update({i: FREE for i in cluster if i not in possible})
    return alloc


def hook_positions(xs: list[float], hook_x: dict[int, float]) -> list[float]:
    """Picture x coordinates -> fractional hook indices, by interpolating between the hooks' centres."""
    idx = sorted(hook_x, key=lambda i: hook_x[i])
    centres = [hook_x[i] for i in idx]
    return [float(np.interp(x, centres, idx, left=idx[0] - (centres[0] - x) / max(1e-6, centres[1] - centres[0]),
                            right=idx[-1] + (x - centres[-1]) / max(1e-6, centres[-1] - centres[-2])))
            for x in xs]
