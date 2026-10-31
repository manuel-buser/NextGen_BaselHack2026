"""1-D projective lattice: a row of equally spaced slots seen in perspective (calibration helper).

Equally spaced world positions X_n = n * P (n = 0, 1, 2, ...) along a straight row do NOT project to a
constant pixel pitch.  Along the image line of the row they form a 1-D homography

    u(n)     = (a*n + b) / (c*n + 1)      b = u(0), a/c = vanishing point of the row, c = foreshortening
    n(u)     = (u - b) / (a - c*u)        continuous lattice index of any point on the line
    pitch(n) = du/dn = (a - b*c) / (c*n + 1)^2

so comparing a gap at the far end of a rack with the median pitch of the whole rack confuses perspective
with "a stand is hidden here".  In lattice units every slot is exactly 1.0 wide, hidden stands sit at the
missing integers, slot boundaries at the half-integers are the perspective-correct ones, and "is that end
bike in an extra slot or just leaning away from the last stand?" becomes a question of whether its index
is nearer to n_max + 1 than to n_max.

fit(): combinatorial integer-gap RANSAC.  For pairs (affine, c = 0) and triplets of neighbouring anchors
and hypothesised integer gaps (1..MAX_GAP) between them the system  n*a + b - (n*u)*c = u  is solved
exactly, every anchor is scored by its distance to the nearest lattice position, and the best hypothesis
is refined by weighted least squares on its inliers.  Strong anchors (stand bases) must sit at integer
indices.  Weak anchors (parked bikes) count half and only have to be *consistent with each other*: bikes
lean on one side of a stirrup, so they sit at integer + phase, and the phase is free.  Lattice positions
between the outermost anchors without any anchor cost a little, so a half-pitch lattice (every second slot
empty) never beats the plain one.  Pure numpy, a few milliseconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

MAX_GAP = 4             # most hidden stands between two detected neighbours in one hypothesis
NEIGHBOURS = 4          # triplets are built from anchors at most this many sorted positions apart
TOL_STRONG = 0.15       # a stand is on the lattice when within this (slot units) of an integer index
TOL_WEAK = 0.30         # a bike is on the lattice when within this of (integer + common phase)
WEIGHT_WEAK = 0.5
EMPTY_PENALTY = 0.2     # score cost of a lattice position between n_min and n_max without any anchor (hidden stand)
MIN_PITCH_RATIO = 0.7   # neighbouring slots may not shrink by more than this factor per slot
MAX_SPAN_RATIO = 5.0    # pixel pitch at one end of the row (plus one slot) vs the other end
WEIGHT_WEAK_SUPPORT = 1.0  # when candidate lattices compete, the bikes are the arbiter and count fully
WEIGHT_DIVIDER = 1.5    # painted dividers are drawn on purpose to mark the bays: they outrank stands that are mounted off-centre
MARGIN = 2              # the lattice must stay regular this many slots beyond the outermost anchor
MAX_EXTRA = 8           # end slots that bikes / stands may add at either end
OVERHANG = 0.65         # a bike further than this (slot units) beyond the last stand is in an extra slot


@dataclass
class Lattice:
    a: float
    b: float
    c: float
    n_min: int = 0
    n_max: int = 0
    strong_in: list[int] = field(default_factory=list)   # indices (into the inputs of fit()) of anchors on the lattice
    weak_in: list[int] = field(default_factory=list)
    weak_phase: float = 0.0                              # bikes sit at integer + weak_phase
    score: float = 0.0
    residual: float = 0.0
    n_left: int = 0                                      # end slots added by extend()
    n_right: int = 0

    # ------------------------------------------------------------------ evaluation
    def u(self, n: float) -> float:
        return (self.a * n + self.b) / (self.c * n + 1.0)

    def n(self, u: float) -> float:
        den = self.a - self.c * u
        return (u - self.b) / den if den > 1e-12 else math.inf

    def pitch(self, n: float) -> float:
        return (self.a - self.b * self.c) / (self.c * n + 1.0) ** 2

    @property
    def count(self) -> int:
        return self.n_max - self.n_min + 1

    def indices(self) -> list[int]:
        return list(range(self.n_min, self.n_max + 1))

    def centres(self) -> list[float]:
        return [self.u(k) for k in self.indices()]

    def bounds(self, phase: float = 0.0) -> list[float]:
        """count + 1 slot boundaries: slot k spans u(k - 0.5 + phase) .. u(k + 0.5 + phase); phase 0 puts
        the stand in the middle of its slot, phase +-0.5 puts the stands ON the boundaries."""
        return [self.u(k - 0.5 + phase) for k in range(self.n_min, self.n_max + 2)]

    def shifted(self, s: float) -> "Lattice":
        """Re-index so that old index n becomes n - s (same positions; s may be fractional)."""
        d = self.c * s + 1.0
        return replace(self, a=self.a / d, b=(self.a * s + self.b) / d, c=self.c / d,
                       n_min=int(round(self.n_min - s)), n_max=int(round(self.n_max - s)))

    def normalized(self) -> "Lattice":
        """Indices 0 .. count-1."""
        return self.shifted(self.n_min)

    def summary(self) -> str:
        return (f"N={self.count} (stands on lattice {len(self.strong_in)}, bikes {len(self.weak_in)}, "
                f"end slots +{self.n_left}/+{self.n_right}), pitch {self.pitch(self.n_min):.0f}->{self.pitch(self.n_max):.0f} px")


def affine(u0: float, pitch: float, n_min: int = 0, n_max: int = 0) -> Lattice:
    """Constant-pitch lattice (c = 0) through u0 at index 0."""
    return Lattice(a=float(pitch), b=float(u0), c=0.0, n_min=n_min, n_max=n_max)


# ---------------------------------------------------------------------- internals

def _indices(a: float, b: float, c: float, us: np.ndarray) -> np.ndarray:
    den = a - c * us
    out = np.full(us.shape, np.inf)
    ok = den > 1e-12
    out[ok] = (us[ok] - b) / den[ok]
    return out


def _frac(ns: np.ndarray, phase: float) -> np.ndarray:
    """Signed distance to the nearest (integer + phase), in [-0.5, 0.5); inf stays inf."""
    with np.errstate(invalid="ignore"):
        out = np.mod(ns - phase + 0.5, 1.0) - 0.5
    out[~np.isfinite(ns)] = np.inf
    return out


def _circular_mean(ns: np.ndarray) -> tuple[float, float]:
    fin = ns[np.isfinite(ns)]
    if len(fin) == 0:
        return 0.0, 0.0
    ang = 2.0 * math.pi * fin
    cs, sn = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    return math.atan2(sn, cs) / (2.0 * math.pi), math.hypot(cs, sn)


def _plausible(a: float, b: float, c: float, n_lo: float, n_hi: float) -> bool:
    """No pole, positive and smoothly varying pitch from MARGIN slots before n_lo to MARGIN after n_hi."""
    k = a - b * c
    if not (k > 0):
        return False
    lo, hi = n_lo - MARGIN, n_hi + MARGIN
    d_lo, d_hi = c * lo + 1.0, c * hi + 1.0
    if d_lo <= 0.05 or d_hi <= 0.05:
        return False
    p_lo, p_hi = k / (c * (n_lo - 1) + 1.0) ** 2, k / (c * (n_hi + 1) + 1.0) ** 2
    if max(p_lo, p_hi) > MAX_SPAN_RATIO * min(p_lo, p_hi):
        return False
    for n in (lo, hi - 1):
        r = ((c * n + 1.0) / (c * (n + 1) + 1.0)) ** 2
        if not (MIN_PITCH_RATIO <= r <= 1.0 / MIN_PITCH_RATIO):
            return False
    return True


def _evaluate(a: float, b: float, c: float, strong: np.ndarray, weak: np.ndarray,
              half: np.ndarray = None, pitch_range=None, weight_weak: float = WEIGHT_WEAK,
              weight_half: float = 1.0) -> Optional[Lattice]:
    """Score one lattice: stands at integers, dividers (``half``) at half-integers, bikes at integer + free
    phase.  Returns None when nothing is on the lattice, it is implausible over its inliers, or its pitch
    leaves ``pitch_range`` (physical bounds from the bike size: a slot is neither narrower than a fraction
    of a bike nor wider than a bike is long)."""
    ns = _indices(a, b, c, strong)
    rs = np.abs(_frac(ns, 0.0))
    s_in = [i for i in range(len(rs)) if rs[i] < TOL_STRONG]
    ks = [int(round(ns[i])) for i in s_in]

    nw = _indices(a, b, c, weak)
    phase_w, _ = _circular_mean(nw)
    rw = np.abs(_frac(nw, phase_w))
    w_in = [i for i in range(len(rw)) if rw[i] < TOL_WEAK]
    kw = [int(round(nw[i] - phase_w)) for i in w_in]

    h_in, kh = [], []
    if half is not None and len(half):
        nh = _indices(a, b, c, half)
        rh = np.abs(_frac(nh, 0.5))
        h_in = [i for i in range(len(rh)) if rh[i] < TOL_STRONG]
        kh = [int(math.floor(nh[i])) for i in h_in]            # divider between slots kh and kh + 1

    if not s_in and not w_in and not h_in:
        return None
    if ks:
        n_min, n_max = min(ks), max(ks)
    elif kw:
        n_min, n_max = min(kw), max(kw)
    else:
        n_min, n_max = min(kh), max(kh) + 1
    all_k = ks + kw + kh + [k + 1 for k in kh]
    if not _plausible(a, b, c, min(all_k + [n_min]), max(all_k + [n_max])):
        return None
    if pitch_range is not None:
        k0 = a - b * c
        for n in (n_min, n_max):
            pn = k0 / (c * n + 1.0) ** 2
            if not (pitch_range[0] <= pn <= pitch_range[1]):
                return None
    filled = set(ks) | {k for k in kw if n_min <= k <= n_max}
    for k in kh:
        filled |= {kk for kk in (k, k + 1) if n_min <= kk <= n_max}
    empty = (n_max - n_min + 1) - len(filled)
    soft_s = sum(1.0 - (rs[i] / TOL_STRONG) ** 2 for i in s_in)
    soft_w = sum(1.0 - (rw[i] / TOL_WEAK) ** 2 for i in w_in)
    soft_h = sum(1.0 - (rh[i] / TOL_STRONG) ** 2 for i in h_in) if h_in else 0.0
    score = soft_s + weight_half * soft_h + weight_weak * soft_w - EMPTY_PENALTY * empty
    residual = (sum((rs[i] / TOL_STRONG) ** 2 for i in s_in)
                + weight_weak * sum((rw[i] / TOL_WEAK) ** 2 for i in w_in))
    lat = Lattice(a, b, c, n_min, n_max, s_in, w_in, phase_w, float(score), float(residual))
    if not s_in and w_in and abs(phase_w) > 1e-9:
        lat = lat.shifted(phase_w)                           # bikes-only lattice: put the bikes ON the integers
        lat.weak_phase = 0.0
    return lat


def _better(x: Lattice, y: Optional[Lattice]) -> bool:
    if y is None:
        return True
    if abs(x.score - y.score) > 1e-9:
        return x.score > y.score
    if x.count != y.count:
        return x.count < y.count            # same support: the lattice with fewer (bigger) slots
    return x.residual < y.residual


def _solve3(ns, us) -> Optional[np.ndarray]:
    A = np.array([[n, 1.0, -n * u] for n, u in zip(ns, us)], dtype=np.float64)
    try:
        x = np.linalg.solve(A, np.asarray(us, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    return x if np.all(np.isfinite(x)) else None


def _hypotheses(pool: np.ndarray, max_gap: int):
    """(a, b, c, n_lo, n_hi) for pairs (affine) and triplets of neighbouring anchors and integer gaps."""
    m = len(pool)
    for i in range(m):
        for j in range(i + 1, min(m, i + 1 + NEIGHBOURS)):
            if pool[j] - pool[i] <= 1e-9:
                continue
            for d1 in range(1, max_gap + 1):
                yield (pool[j] - pool[i]) / d1, pool[i], 0.0, 0, d1
            for k in range(j + 1, min(m, j + 1 + NEIGHBOURS)):
                if pool[k] - pool[j] <= 1e-9:
                    continue
                for d1 in range(1, max_gap + 1):
                    for d2 in range(1, max_gap + 1):
                        x = _solve3([0, d1, d1 + d2], [pool[i], pool[j], pool[k]])
                        if x is not None:
                            yield float(x[0]), float(x[1]), float(x[2]), 0, d1 + d2


def _refine(lat: Lattice, strong: np.ndarray, weak: np.ndarray, iterations: int = 2, pitch_range=None) -> Lattice:
    for _ in range(iterations):
        ns, nw = _indices(lat.a, lat.b, lat.c, strong), _indices(lat.a, lat.b, lat.c, weak)
        rows, rhs, wts = [], [], []
        for i in lat.strong_in:
            n = round(ns[i])
            rows.append([n, 1.0, -n * strong[i]]); rhs.append(strong[i]); wts.append(1.0)
        for i in lat.weak_in:
            n = round(nw[i] - lat.weak_phase) + lat.weak_phase
            rows.append([n, 1.0, -n * weak[i]]); rhs.append(weak[i]); wts.append(WEIGHT_WEAK)
        if len(rows) < 3:
            break
        sw = np.sqrt(np.asarray(wts))[:, None]
        x, *_ = np.linalg.lstsq(np.asarray(rows) * sw, np.asarray(rhs) * sw[:, 0], rcond=None)
        if not np.all(np.isfinite(x)):
            break
        cand = _evaluate(float(x[0]), float(x[1]), float(x[2]), strong, weak, pitch_range=pitch_range)
        if cand is None or cand.score < lat.score - 1e-9:
            break
        if abs(cand.score - lat.score) < 1e-9 and cand.residual >= lat.residual:
            break
        lat = cand
    return lat


def _rescale(lat: Lattice, m: float, k: float) -> Lattice:
    """Lattice fitted on (u - m) / k -> lattice in u."""
    return replace(lat, a=k * lat.a + m * lat.c, b=k * lat.b + m)


# ---------------------------------------------------------------------- public API

def fit(strong: Sequence[float], weak: Sequence[float] = (), max_gap: int = MAX_GAP,
        pitch_range: Optional[tuple[float, float]] = None) -> Optional[Lattice]:
    """Fit the lattice to 1-D axis coordinates of stand bases (``strong``) and parked bikes (``weak``).

    Needs at least 2 anchors in total; with only 2 the lattice is affine.  ``pitch_range`` bounds the pixel
    pitch at the ends of the row (from the bike size).  Returns None when nothing plausible fits.  Indices
    in the result refer to the order of the inputs; n_min..n_max is the index range covered by stands (or
    by bikes when there are no stands on the lattice).
    """
    s = np.asarray(strong, dtype=np.float64).ravel()
    w = np.asarray(weak, dtype=np.float64).ravel()
    both = np.concatenate([s, w])
    if len(both) < 2:
        return None
    m = float(both.min())
    k = max(float(np.ptp(both)), 1.0)
    sn, wn = (s - m) / k, (w - m) / k                     # unit-free coordinates: well-conditioned solves
    pr = None if pitch_range is None else (pitch_range[0] / k, pitch_range[1] / k)
    pools = []
    if len(sn) >= 2:
        pools.append(np.sort(sn))
    if len(wn) >= 3 or (len(sn) < 2 and len(wn) >= 2):
        pools.append(np.sort(wn))
    if len(sn) < 3 and len(sn) + len(wn) >= 3:
        pools.append(np.sort(np.concatenate([sn, wn])))
    best: Optional[Lattice] = None
    seen: set = set()
    for pool in pools:
        for a, b, c, lo, hi in _hypotheses(pool, max_gap):
            if not _plausible(a, b, c, lo, hi):
                continue
            key = (round(a, 7), round(b, 7), round(c, 7))
            if key in seen:
                continue
            seen.add(key)
            lat = _evaluate(a, b, c, sn, wn, pitch_range=pr)
            if lat is not None and _better(lat, best):
                best = lat
    if best is None:
        return None
    return _rescale(_refine(best, sn, wn, pitch_range=pr), m, k)


def support(lat: Lattice, strong: Sequence[float] = (), half: Sequence[float] = (),
            weak: Sequence[float] = ()) -> tuple[float, str]:
    """How well ``lat`` explains all the evidence: stands at integers, painted dividers at half-integers,
    bikes at integer + common phase (same scoring as fit(), but the bikes count fully: a slot layout has
    to explain where the bikes park).  Returns (score, breakdown text); score -inf when nothing fits."""
    s = np.asarray(strong, dtype=np.float64).ravel()
    w = np.asarray(weak, dtype=np.float64).ravel()
    hh = np.asarray(half, dtype=np.float64).ravel()
    cand = _evaluate(lat.a, lat.b, lat.c, s, w, hh, weight_weak=WEIGHT_WEAK_SUPPORT, weight_half=WEIGHT_DIVIDER)
    if cand is None:
        return -math.inf, "nothing on the lattice"
    nh = _indices(lat.a, lat.b, lat.c, hh)
    n_half = int(np.sum(np.abs(_frac(nh, 0.5)) < TOL_STRONG)) if len(hh) else 0
    text = f"{len(cand.strong_in)}/{len(s)} stands, {n_half}/{len(hh)} dividers, {len(cand.weak_in)}/{len(w)} bikes"
    return cand.score, text


def extend(lat: Lattice, us: Sequence[float], bias: float = 0.0, overhang: float = OVERHANG,
           max_extra: int = MAX_EXTRA, strong_us: Sequence[float] = ()) -> Lattice:
    """Add the end slots implied by bikes (``us``) or stands (``strong_us``) beyond the outermost anchors
    of the lattice.

    A bike whose index (minus ``bias``, the typical offset of bikes from their stand) lies more than
    ``overhang`` beyond n_max occupies slot n_max + 1 (its stand is hidden behind it); closer than that it
    is leaning away from the last slot.  A stand beyond the range is a slot wherever it rounds to.
    """
    left = right = 0
    for u, off, ovh in [(x, bias, overhang) for x in us] + [(x, 0.0, 0.5) for x in strong_us]:
        nh = lat.n(float(u)) - off
        if not math.isfinite(nh):
            continue
        if nh > lat.n_max + ovh:
            right = max(right, max(1, int(round(nh - lat.n_max))))
        elif nh < lat.n_min - ovh:
            left = max(left, max(1, int(round(lat.n_min - nh))))
    left, right = min(left, max_extra), min(right, max_extra)
    while right and not _plausible(lat.a, lat.b, lat.c, lat.n_min, lat.n_max + right):
        right -= 1
    while left and not _plausible(lat.a, lat.b, lat.c, lat.n_min - left, lat.n_max):
        left -= 1
    return replace(lat, n_min=lat.n_min - left, n_max=lat.n_max + right,
                   n_left=lat.n_left + left, n_right=lat.n_right + right)


def phase(lat: Lattice, us: Sequence[float]) -> tuple[float, float]:
    """Circular mean of the fractional lattice index of ``us`` -> (phase in [-0.5, 0.5), concentration 0..1).

    Painted bay dividers at phase +-0.5 mean the stands are centred in their bays; bikes at phase 0.3
    mean bikes typically stand 0.3 slot to the right of their stand (useful as ``bias`` for extend()).
    """
    return _circular_mean(np.asarray([lat.n(float(u)) for u in us], dtype=np.float64))
