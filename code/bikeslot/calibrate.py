"""Stage A: propose a slot layout for a camera from its buffered frames (Layout(status="proposed") for the admin).

Anchors (stands, parked bikes, painted dividers, optional VLM hints) are projected onto each row axis and a 1-D
projective lattice (lattice.py) is fitted, so hidden stands and extra end slots are decided in slot units, not pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import cv2
import numpy as np

from . import config, detector, lattice, stands, vlm
from .geometry import normalize, pixel_polygon, point_in_polygon, slots_from_rows
from .preview import blur_boxes
from .schemas import CalibrationHint, Detection, Layout, Row, utcnow

FRONT_PAD_FRAC = 0.08      # front edge pushed towards the aisle by this share of the bike height
BACK_PAD_FRAC = 0.05       # back edge pushed behind the stand bases by this share of the bike height
MIN_CONF_FOR_LAYOUT = 0.3  # detections below this are ignored for calibration
ROW_SPLIT_FRAC = 0.45      # contact points further apart (across the row) than this share of the bike height = new row
MIN_LATTICE_ANCHORS = 3    # stands / dividers on the lattice for its count to override the other estimates
MIN_DIVIDERS = 4           # painted dividers needed for a lattice of their own (3 points always fit exactly)
STAND_AT_FAR_END = 0.4     # stands further than this share of the bike height from the rear wheels = at the front wheel
FAR_SIDE_FRAC = 0.3        # a bike whose centre is this share of its height beyond the stand line is parked on the far side
SIDE_GAP_FRAC = 0.15       # the two groups must be separated by this share of the bike height to count as two sides
DIVIDER_PHASE_CONC = 0.7   # dividers must agree this well (circular concentration) to place the boundaries
PITCH_RANGE_H = (0.15, 1.2)  # slot pitch relative to the bike height (physical bounds for the lattice fit)


# =========================================================================== entry point

def propose_layout(cam, use_vlm: bool = True, use_stands: bool = True) -> Layout:
    """``cam`` is a store.CameraState (duck-typed to avoid an import cycle)."""
    frames = [f for _, f in cam.frames]
    if not frames:
        raise ValueError("no frames buffered for this camera")
    h, w = frames[-1].shape[:2]
    frame_size = (w, h)
    frames = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]

    dets_pf = [detector.detect(f) for f in frames]
    vehicles_pf = [[d for d in ds if d.cls in config.VEHICLE_CLASSES and d.conf >= MIN_CONF_FOR_LAYOUT] for ds in dets_pf]
    bikes = [_bike(d, (h, w), fi) for fi, vs in enumerate(vehicles_pf) for d in vs]
    per_frame = [len(vs) for vs in vehicles_pf]

    st = stands.detect_stands(frames[-5:], vehicles_pf[-5:]) if use_stands else None
    anchors = st["anchors"] if st else []

    hint: Optional[CalibrationHint] = None
    mark_points: list[list[float]] = []
    if use_vlm:
        last, last_dets = frames[-1], dets_pf[-1]
        marks = sorted((i for i, d in enumerate(last_dets) if d in vehicles_pf[-1]),
                       key=lambda i: (last_dets[i].contact or last_dets[i].bottom_center)[0])
        img = vlm.mark_frame(blur_boxes(last, last_dets, inplace=False), last_dets, marks)
        if img.shape[1] > config.VLM_MAX_SIDE:
            s = config.VLM_MAX_SIDE / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            hint = vlm.describe_scene(buf.tobytes(), n_marks=len(marks))
        mark_points = [list(last_dets[i].contact or last_dets[i].bottom_center) for i in marks]

    painted = None
    if bikes:
        painted = painted_dividers(frames, bikes, frame_size)

    layout = fuse(cam.camera_id, frame_size, bikes, anchors, hint, mark_points, painted,
                  n_frames=len(frames), max_per_frame=max(per_frame) if per_frame else 0,
                  last_frame_vehicles=per_frame[-1] if per_frame else 0,
                  stand_notes=st["notes"] if st else None)
    layout.site_id = cam.layout.site_id
    layout.site_name = cam.layout.site_name
    return layout


# =========================================================================== bikes

def _bike(det: Detection, shape: tuple[int, int], frame_index: int = 0) -> dict:
    """Contact point (rear wheel, on the ground), far end (top of the mask) and size of one detected bike."""
    contact = list(det.contact or det.bottom_center)
    far = None
    if det.mask:
        ys, xs = detector.mask_pixels(det, shape)
        if len(ys):
            band = max(3, int(0.10 * det.height))
            sel = ys <= ys.min() + band
            far = [float(xs[sel].mean()), float(ys[sel].mean())]
    if far is None:
        far = [(det.box[0] + det.box[2]) / 2.0, det.box[1]]
    return {"contact": contact, "far": far, "w": det.width, "h": det.height, "frame": frame_index}


def _mid(b: dict) -> list[float]:
    """Centre of the bike (between rear-wheel contact and far end): the point least affected by leaning."""
    return [(b["contact"][0] + b["far"][0]) / 2.0, (b["contact"][1] + b["far"][1]) / 2.0]


def split_rows(bikes: list[dict]) -> list[list[dict]]:
    """Group bikes into rows by the height of their contact point.  A second row is roughly a bike height
    further up in the image; the threshold uses the smaller of the two neighbouring bikes so that a far row
    of small bikes is still separated from a near row of big ones, while a diagonal row (contact height
    changing gradually) stays one row.  Rows are returned front (largest y) first."""
    if not bikes:
        return []
    pts = np.array([b["contact"] for b in bikes], dtype=np.float64)
    hs = np.array([b["h"] for b in bikes], dtype=np.float64)
    remaining = list(range(len(bikes)))
    rows: list[list[int]] = []
    # sequential line fitting: the line through the most contact points is a row, remove them, repeat
    while len(remaining) >= 2:
        best: tuple[int, list[int]] = (0, [])
        for a in range(len(remaining)):
            for b in range(a + 1, len(remaining)):
                i, j = remaining[a], remaining[b]
                d = pts[j] - pts[i]
                norm = np.hypot(*d)
                if norm < 1e-6:
                    continue
                nrm = np.array([-d[1], d[0]]) / norm
                inliers = [k for k in remaining
                           if abs(float(np.dot(pts[k] - pts[i], nrm))) <= ROW_SPLIT_FRAC * float(min(hs[k], hs[i], hs[j]))]
                if len(inliers) > best[0]:
                    best = (len(inliers), inliers)
        if best[0] < 2:
            break
        rows.append(best[1])
        remaining = [k for k in remaining if k not in best[1]]
    for k in remaining:                        # leftovers join the row whose line is nearest
        if not rows:
            rows.append([k])
            continue
        dists = []
        for r in rows:
            rp = pts[r]
            if len(rp) >= 2:
                a_, b_ = _line_fit(rp.tolist())
                dists.append(abs(a_ * pts[k][0] - pts[k][1] + b_) / np.hypot(a_, 1.0))
            else:
                dists.append(float(np.hypot(*(pts[k] - rp[0]))))
        rows[int(np.argmin(dists))].append(k)
    out = [[bikes[i] for i in r] for r in rows]
    out.sort(key=lambda r: -float(np.mean([b["contact"][1] for b in r])))
    return out


def max_per_frame_in_row(row: list[dict]) -> int:
    counts: dict[int, int] = {}
    for b in row:
        counts[b["frame"]] = counts.get(b["frame"], 0) + 1
    return max(counts.values()) if counts else 0


# =========================================================================== painted dividers

def painted_dividers(frames: list[np.ndarray], bikes: list[dict], frame_size: tuple[int, int]) -> Optional[dict]:
    """Look for painted bay dividers inside a generous quad around the (front row of) bikes (lines.py)."""
    from . import lines

    front = split_rows(bikes)[0]
    w, h = frame_size
    pts = np.array([b["contact"] for b in front], dtype=np.float64)
    med_h = float(np.median([b["h"] for b in front]))
    med_w = float(np.median([b["w"] for b in front]))
    a, b = _line_fit(pts.tolist())
    left = max(0.0, float(pts[:, 0].min()) - 1.5 * med_w)
    right = min(float(w - 1), float(pts[:, 0].max()) + 1.5 * med_w)

    def y_at(x: float, dy: float) -> float:
        return float(min(h - 1, max(0.0, a * x + b + dy)))

    quad = [[left, y_at(left, -1.1 * med_h)], [right, y_at(right, -1.1 * med_h)],
            [right, y_at(right, 0.15 * med_h)], [left, y_at(left, 0.15 * med_h)]]
    try:
        return lines.find_dividers(frames[-5:], quad)
    except Exception as exc:  # never let the refinement break calibration
        print(f"[calibrate] divider detection failed: {exc}")
        return None


# =========================================================================== VLM marks

def marks_consistent(hint: CalibrationHint, mark_points: list[list[float]], pitch: float) -> tuple[bool, str]:
    """Is the VLM's 'bike k is in slot j' answer consistent with the bikes' positions?  Marks are numbered
    left->right, so the index difference between neighbouring marked bikes must match their distance in
    pitches (an empty bay between two bikes means the indices differ by 2).  The VLM often just numbers
    the bikes 1..k, which this catches."""
    if not hint.mark_slots or pitch <= 0:
        return False, "no marks or no pitch"
    seq = [(mark_points[m - 1][0], j) for m, j in sorted(hint.mark_slots.items())
           if 1 <= m <= len(mark_points) and j >= 1]
    if len(seq) < 2:
        return False, "fewer than 2 marked bikes in the row"
    seq.sort()
    ok = total = 0
    for (x0, j0), (x1, j1) in zip(seq[:-1], seq[1:]):
        total += 1
        expected = max(1, int(round((x1 - x0) / pitch)))
        ok += (j1 - j0) == expected
    good = ok >= max(1, int(np.ceil(0.75 * total)))
    return good, f"{ok}/{total} neighbour index gaps match the spacing"


# =========================================================================== geometry helpers

def _line_fit(points) -> tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= 2 and np.ptp(pts[:, 0]) > 1.0:
        a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
        return float(a), float(b)
    return 0.0, float(pts[:, 1].mean())


def _axis(points) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction (x increasing) and a point on the least-squares line through ``points``."""
    a, b = _line_fit(points)
    u = np.array([1.0, a]) / np.hypot(1.0, a)
    return u, np.array([0.0, b])


def _assign_anchors_to_rows(anchors: list[dict], rows: list[list[dict]]) -> list[list[dict]]:
    """Each stand base belongs to the row whose expected back edge (contact y - ~0.5 bike height) is nearest."""
    if not rows:
        return [list(anchors)]
    targets = []
    for r in rows:
        med_h = float(np.median([b["h"] for b in r]))
        targets.append(float(np.median([b["contact"][1] for b in r])) - 0.5 * med_h)
    out: list[list[dict]] = [[] for _ in rows]
    for a in anchors:
        k = int(np.argmin([abs(a["y"] - t) for t in targets]))
        out[k].append(a)
    return out


def _merge_far_side_rows(rows_b: list[list[dict]], rows_a: list[list[dict]]) -> tuple[list, list, int]:
    """Merge a few stand-less bikes just behind another row's stand line (within ~a bike height) into that row as
    its far side. A row as full as the stand row stays its own (facing racks across an aisle look the same)."""
    rows_b, rows_a = [list(r) for r in rows_b], [list(r) for r in rows_a]
    merged = 0
    changed = True
    while changed:
        changed = False
        for i, (rb, ra) in enumerate(zip(rows_b, rows_a)):
            if len(ra) >= 2 or not rb:
                continue
            for j, (rb2, ra2) in enumerate(zip(rows_b, rows_a)):
                if j == i or len(ra2) < 2:
                    continue
                u, p0 = _axis([[a["x"], a["y"]] for a in ra2])
                v = np.array([-u[1], u[0]])
                if v[1] < 0:
                    v = -v
                d_i = float(np.median([np.dot(np.asarray(b["contact"]) - p0, v) for b in rb]))
                d_j = float(np.median([np.dot(np.asarray(b["contact"]) - p0, v) for b in rb2])) if rb2 else 1.0
                med_h = float(np.median([b["h"] for b in rb]))
                if d_i < 0 and d_j > 0 and -d_i < 1.2 * med_h and len(rb) < len(rb2):
                    rows_b[j] = rb2 + rb
                    rows_b.pop(i)
                    rows_a.pop(i)
                    merged += 1
                    changed = True
                    break
            if changed:
                break
    return rows_b, rows_a, merged


def end_slots_from_bikes(bikes: list[dict], anchors: list[dict]) -> tuple[int, int]:
    """Affine fallback: bikes parked clearly beyond the outermost stands (along the row axis) imply extra
    end slots.  The lattice version (lattice.extend) is used whenever a lattice could be fitted."""
    if len(anchors) < 2 or not bikes:
        return 0, 0
    u, p0 = _axis([[a["x"], a["y"]] for a in anchors])
    at = sorted(float(np.dot(np.array([a["x"], a["y"]]) - p0, u)) for a in anchors)
    pitch = stands.pitch_estimate(at)
    if pitch <= 1.0:
        return 0, 0
    tb = [float(np.dot(np.array(b["far"]) - p0, u)) for b in bikes]   # far end of the bike ~ at its stand
    n_left = max([int(np.floor((at[0] - t) / pitch + 0.4)) for t in tb] + [0])
    n_right = max([int(np.floor((t - at[-1]) / pitch + 0.4)) for t in tb] + [0])
    return min(n_left, 3), min(n_right, 3)


# =========================================================================== lattice row model

@dataclass
class DepthModel:
    """One side of the row: the back line of its slots (the stand line when the stands sit at the front
    wheels, else the line through the far ends of the bikes) and the depth vector from that line to the
    rear-wheel contacts.  The depth scales with the local lattice pitch (both shrink with distance) and may
    turn slowly along the row (perspective)."""
    sign: int                                  # +1: bikes on the camera side of the stand line, -1: far side
    pb: np.ndarray = None                      # back line: point + unit direction
    ub: np.ndarray = None
    bias: float = 0.0                          # lattice boundaries on the back line are shifted by this (slot units)
    rho: float = 0.0                           # |depth| / pitch at t_ref
    drho: float = 0.0                          # change of rho per pixel along the axis (fitted when bikes span the row)
    rho_lo: float = 0.0                        # rho is clipped to [rho_lo, rho_hi] when extrapolating
    rho_hi: float = 1e9
    theta: float = 0.0                         # direction at t_ref
    dtheta: float = 0.0                        # direction change per pixel along the axis
    t_ref: float = 0.0
    const: Optional[np.ndarray] = None         # fixed vector when no bike gives a depth
    n_bikes: int = 0
    note: str = ""

    def vector(self, t: float, lat: lattice.Lattice) -> np.ndarray:
        if self.const is not None:
            return self.const
        th = self.theta + self.dtheta * (t - self.t_ref)
        rho = min(self.rho_hi, max(self.rho_lo, self.rho + self.drho * (t - self.t_ref)))
        return rho * lat.pitch(lat.n(t)) * np.array([math.cos(th), math.sin(th)])


@dataclass
class RowModel:
    lattice: lattice.Lattice
    source: str                                # "stands", "dividers" or "bikes"
    n_anchors: int                             # stands / dividers on the lattice (0 for a bikes-only lattice)
    u: np.ndarray                              # unit axis direction (x increasing)
    p0: np.ndarray                             # origin of the axis coordinate t
    v: np.ndarray                              # unit normal, pointing down the image (towards the camera)
    ub: np.ndarray                             # back line direction / point (stand bases)
    pb: np.ndarray
    med_h: float
    bias: float = 0.0                          # typical offset of bike centres from their stand, slot units
    slot_phase: float = 0.0                    # slot boundaries at u(k - 0.5 + slot_phase)
    bike_slots: list = field(default_factory=list)
    sides: dict = field(default_factory=dict)  # sign -> DepthModel
    notes: list = field(default_factory=list)

    def t(self, p) -> float:
        return float(np.dot(np.asarray(p, dtype=np.float64) - self.p0, self.u))

    def back_at(self, t: float, pb: Optional[np.ndarray] = None, ub: Optional[np.ndarray] = None) -> np.ndarray:
        """Point of the (stand) line whose projection on the axis is t; another line may be given."""
        pb = self.pb if pb is None else pb
        ub = self.ub if ub is None else ub
        s = (t - float(np.dot(pb - self.p0, self.u))) / max(float(np.dot(ub, self.u)), 0.5)
        return pb + ub * s

    @property
    def count(self) -> int:
        return self.lattice.count

    @property
    def bilateral(self) -> bool:
        return len(self.sides) == 2


def row_model(rb: list[dict], ra: list[dict], dividers: Optional[list], frame_size: tuple[int, int]) -> Optional[RowModel]:
    """Axis, lattice, bike->slot assignment and depth model(s) for one row.  None when there is nothing to
    fit a lattice to (the caller then uses the affine row_geometry)."""
    notes: list[str] = []
    stand_pts = [[a["x"], a["y"]] for a in ra]
    contacts = [b["contact"] for b in rb]
    if len(stand_pts) >= 2:
        u, p0 = _axis(stand_pts)
        back_pts = stand_pts
        notes.append("axis from stands")
    elif len(contacts) >= 2:
        u, p0 = _axis(contacts)
        back_pts = [b["far"] for b in rb]
        notes.append("axis from bikes")
    else:
        return None
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v
    ub, pb = _axis(back_pts) if len(back_pts) >= 2 else (u, np.array(back_pts[0], dtype=np.float64))
    med_h = float(np.median([b["h"] for b in rb])) if rb else 3.0 * float(np.median([a["h"] for a in ra]))

    def t_of(p) -> float:
        return float(np.dot(np.asarray(p, dtype=np.float64) - p0, u))

    ts_stand = [t_of(p) for p in stand_pts]
    ts_bike = [t_of(_mid(b)) for b in rb]
    ts_div = [t_of(p) for p in dividers] if dividers else []

    # ---- candidate lattices: stands, painted dividers, bikes; the one explaining most evidence wins.
    # A candidate's own anchors are discounted by the 3 parameters they were fitted with.
    pitch_range = (PITCH_RANGE_H[0] * med_h, PITCH_RANGE_H[1] * med_h)
    cands: list[tuple[str, int, lattice.Lattice, float]] = []
    if len(ts_stand) >= 2:
        L = lattice.fit(ts_stand, ts_bike, pitch_range=pitch_range)
        if L is not None and len(L.strong_in) >= 2:
            cands.append(("stands", len(L.strong_in), L, 3.0))
    if len(ts_div) >= MIN_DIVIDERS:
        Ld = lattice.fit(ts_div, pitch_range=pitch_range)
        if Ld is not None and len(Ld.strong_in) >= MIN_DIVIDERS:
            Lc = Ld.shifted(0.5)                     # dividers at integers -> bay centres at integers
            Lc.n_min, Lc.n_max = Ld.n_min, Ld.n_max - 1
            Lc.strong_in, Lc.weak_in = [], []
            cands.append(("dividers", len(Ld.strong_in), Lc, 3.0))
    if len(ts_bike) >= 3:
        Lb = lattice.fit([], ts_bike, pitch_range=pitch_range)
        if Lb is not None:
            cands.append(("bikes", 0, Lb, 3.0 * lattice.WEIGHT_WEAK_SUPPORT))
    if not cands:
        return None
    scored = []
    for i, (src, n_anch, L, discount) in enumerate(cands):
        sc, text = lattice.support(L, ts_stand, ts_div, ts_bike)
        scored.append((sc - discount, -i, src, n_anch, L, text))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    _, _, source, n_anchors, L, _ = scored[0]
    notes.append("lattice from " + source + " (" + "; ".join(f"{s[2]}: {s[0]:.1f} = {s[5]}" for s in scored) + ")")

    # ---- bikes relative to the lattice: common offset, end slots, slot per bike
    inside = [x for x in ts_bike if L.n_min - 0.5 <= L.n(x) <= L.n_max + 0.5]
    bias, conc = lattice.phase(L, inside) if len(inside) >= 2 else (0.0, 0.0)
    if conc < 0.5:
        bias = 0.0
    L = lattice.extend(L, ts_bike, bias=bias, strong_us=ts_stand if source != "stands" else ())
    if L.n_left or L.n_right:
        notes.append(f"bikes/stands add {L.n_left} end slot(s) left, {L.n_right} right")
    bike_slots = []
    for x in ts_bike:
        n = L.n(x) - bias
        bike_slots.append(int(min(L.n_max, max(L.n_min, round(n)))) if math.isfinite(n) else None)

    slot_phase = 0.0
    if source == "stands" and len(ts_div) >= 3:
        psi, c2 = lattice.phase(L, ts_div)
        if c2 >= DIVIDER_PHASE_CONC:
            slot_phase = psi + 0.5 if psi < 0 else psi - 0.5
            notes.append(f"painted dividers at phase {psi:+.2f} -> slot boundaries shifted by {slot_phase:+.2f}")

    model = RowModel(L, source, n_anchors, u, p0, v, ub, pb, med_h, bias, slot_phase, bike_slots, {}, notes)

    # ---- which side of the stand line is each bike on (by its centre, relative to its height).  A rack is
    # used from both sides only when at least two bikes sit clearly behind the stand line AND there is a
    # clear gap to the bikes in front; otherwise all bikes form one side.
    recs = []
    for b, k in zip(rb, bike_slots):
        if k is None:
            continue
        tk = L.u(k)
        off = float(np.dot(np.asarray(_mid(b)) - model.back_at(tk), v)) / max(b["h"], 1.0)
        recs.append((tk, b, off, k))
    behind = [r for r in recs if r[2] < -FAR_SIDE_FRAC]
    front = [r for r in recs if r[2] >= -FAR_SIDE_FRAC]
    bilateral = (len(behind) >= 2 and len(front) >= 2
                 and min(r[2] for r in front) - max(r[2] for r in behind) >= SIDE_GAP_FRAC)
    if bilateral:
        groups = {1: front, -1: behind}
    elif recs:
        sign = 1 if float(np.median([r[2] for r in recs])) >= -FAR_SIDE_FRAC else -1
        groups = {sign: recs}
    else:
        groups = {}
    for s, grp in groups.items():
        dm = _depth_model(model, [(t, b, s, k) for t, b, _, k in grp], s, L, with_stands=len(ra) >= 2)
        if dm is not None:
            model.sides[s] = dm
    if not model.sides:
        if ra:
            stand_h = float(np.median([a["h"] for a in ra]))
            model.sides[1] = DepthModel(1, pb=pb, ub=ub, const=v * config.SLOT_DEPTH_PER_STAND_HEIGHT * stand_h)
            notes.append("no bike gives the depth: slot depth from the stand height")
        else:
            model.sides[1] = DepthModel(1, pb=pb, ub=ub, const=v * 0.8 * med_h)
    for s, dm in model.sides.items():
        if dm.note:
            notes.append(dm.note)
    if model.bilateral:
        notes.append(f"rack used from both sides ({len(front)} bikes in front, {len(behind)} behind)")
    return model


def _depth_model(model: RowModel, grp: list, sign: int, L: lattice.Lattice, with_stands: bool) -> Optional[DepthModel]:
    """Back line + depth vectors for the bikes of one side.  Stands sitting at the far end of the bikes
    (stirrups at the front wheel) are the back line; otherwise (wheel-holder rails detected near the rear
    wheels, or no stands at all) the line through the far ends of the bikes is."""
    ts = np.array([r[0] for r in grp])
    contacts = np.array([r[1]["contact"] for r in grp], dtype=np.float64)
    fars = np.array([r[1]["far"] for r in grp], dtype=np.float64)
    hs = np.array([r[1]["h"] for r in grp], dtype=np.float64)
    stand_pts = np.array([model.back_at(t) for t in ts])
    depth_stand = np.array([float(np.dot(c - sp, model.v)) * sign for c, sp in zip(contacts, stand_pts)])
    note = ""
    if with_stands and float(np.median(depth_stand / hs)) >= STAND_AT_FAR_END:
        pb, ub, bias = model.pb, model.ub, 0.0
        ds = contacts - stand_pts
    else:
        if len(grp) >= 2:
            ub, pb = _axis(fars.tolist())
        else:
            ub, pb = model.u, fars[0]
        ds = contacts - fars
        nf = np.array([L.n(model.t(f)) for f in fars]) - np.array([r[3] for r in grp])
        bias = float(lattice._circular_mean(nf)[0]) if len(grp) >= 2 else 0.0
        side_name = "front" if sign > 0 else "back"
        note = f"side {side_name}: back line through the far ends of the bikes"
    lengths = np.linalg.norm(ds, axis=1)
    keep = lengths > 0.15 * hs
    if not np.any(keep):
        return None
    ts, ds, lengths = ts[keep], ds[keep], lengths[keep]
    ks = np.array([r[3] for r in grp])[keep]
    units = ds / lengths[:, None]
    mean_dir = units.mean(axis=0)
    theta = float(math.atan2(mean_dir[1], mean_dir[0]))
    rhos = np.array([ln / max(L.pitch(k), 1e-6) for ln, k in zip(lengths, ks)])
    dm = DepthModel(sign, pb=np.asarray(pb, dtype=np.float64), ub=np.asarray(ub, dtype=np.float64), bias=bias,
                    rho=float(np.median(rhos)), rho_lo=0.5 * float(rhos.min()), rho_hi=2.0 * float(rhos.max()),
                    theta=theta, t_ref=float(ts.mean()), n_bikes=int(keep.sum()), note=note)
    pitch_mid = L.pitch(0.5 * (L.n_min + L.n_max))
    if len(ts) >= 3 and float(np.ptp(ts)) > 2.0 * pitch_mid:
        # the slot depth need not shrink exactly like the pitch: follow the bikes along the row
        slope, icpt = np.polyfit(ts - ts.mean(), rhos, 1)
        dm.rho, dm.drho = float(icpt), float(slope)
    if len(ts) >= 4 and float(np.ptp(ts)) > 3.0 * pitch_mid:
        angles = np.arctan2(units[:, 1], units[:, 0])
        angles = theta + np.angle(np.exp(1j * (angles - theta)))     # unwrap around the mean
        slope = float(np.polyfit(ts - ts.mean(), angles, 1)[0])
        dm.dtheta = float(np.clip(slope, -1e-3, 1e-3))                # at most ~1 rad per 1000 px
    return dm


def row_polygons(model: RowModel, frame_size: tuple[int, int]) -> dict[int, list[list[list[float]]]]:
    """Slot quadrilaterals per side (sign -> list of n quads, vertex order back-left, back-right, front-right,
    front-left; for the far side of a bilateral rack 'back' is the side away from the stands)."""
    w, h = frame_size
    L = model.lattice

    def clamp(p) -> list[float]:
        return [float(min(w - 1, max(0.0, p[0]))), float(min(h - 1, max(0.0, p[1])))]

    out: dict[int, list] = {}
    for sign, dm in model.sides.items():
        bounds = L.bounds(model.slot_phase + dm.bias)
        polys = []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            quad = []
            for tt in (t0, t1):
                base = model.back_at(tt, dm.pb, dm.ub)
                d = dm.vector(tt, L)
                unit = d / max(float(np.linalg.norm(d)), 1e-9)
                b = base - unit * BACK_PAD_FRAC * model.med_h
                f = base + d + unit * FRONT_PAD_FRAC * model.med_h
                quad.append((clamp(b), clamp(f)))
            (b0, f0), (b1, f1) = quad
            # first two vertices = the edge further from the camera (higher in the image)
            polys.append([b0, b1, f1, f0] if (f0[1] + f1[1]) >= (b0[1] + b1[1]) else [f0, f1, b1, b0])
        out[sign] = polys
    return out


def _rerange(L: lattice.Lattice, n: int, bike_slots: list) -> lattice.Lattice:
    """Adjust the lattice range to n slots: drop empty end slots first, else add slots at the ends."""
    used = {k for k in bike_slots if k is not None}
    while L.count > n:
        if L.n_max not in used:
            L = replace(L, n_max=L.n_max - 1)
        elif L.n_min not in used:
            L = replace(L, n_min=L.n_min + 1)
        else:
            L = replace(L, n_max=L.n_max - 1)
    flip = True
    while L.count < n:
        L = replace(L, n_max=L.n_max + 1) if flip else replace(L, n_min=L.n_min - 1)
        flip = not flip
    return L


# =========================================================================== affine fallback

def row_geometry(bikes: list[dict], anchors: list[dict], n: int, frame_size: tuple[int, int],
                 n_left: int = 0, n_right: int = 0, corners_hint=None) -> tuple[list, list, str, Optional[list]]:
    """Affine fallback for rows where no lattice fits (a single bike / stand).  Returns (front [P0,P1],
    back [Q0,Q1], note, None) in pixels; the caller splits the row equally."""
    w, h = frame_size
    contacts = np.array([b["contact"] for b in bikes], dtype=np.float64) if bikes else np.zeros((0, 2))
    med_h = float(np.median([b["h"] for b in bikes])) if bikes else (float(np.median([a["h"] for a in anchors])) * 3.0)

    if len(anchors) >= 2:
        u, p0 = _axis([[a["x"], a["y"]] for a in anchors])
        note = "axis from stands"
    elif len(contacts) >= 2:
        u, p0 = _axis(contacts.tolist())
        note = "axis from bikes"
    elif corners_hint is not None:
        c = np.array(corners_hint, dtype=np.float64)
        return [c[3].tolist(), c[2].tolist()], [c[0].tolist(), c[1].tolist()], "row corners from the VLM", None
    else:
        u, p0 = np.array([1.0, 0.0]), np.array([0.0, contacts[0, 1] if len(contacts) else h * 0.6])
        note = "axis horizontal (single bike)"
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v

    def t_of(p):
        return float(np.dot(np.asarray(p) - p0, u))

    if bikes:
        dmed = np.median(np.array([np.array(b["contact"]) - np.array(b["far"]) for b in bikes]), axis=0)
        if anchors:
            a0 = anchors[int(np.argmin([abs(t_of([a["x"], a["y"]]) - t_of(bikes[0]["far"])) for a in anchors]))]
            dmed = np.array(bikes[0]["contact"]) - np.array([a0["x"], a0["y"]])
        depth = dmed
    else:
        depth = v * config.SLOT_DEPTH_PER_STAND_HEIGHT * float(np.median([a["h"] for a in anchors]))

    back_pts = [[a["x"], a["y"]] for a in anchors] if len(anchors) >= 2 else \
        [(np.array(b["contact"]) - depth).tolist() for b in bikes]
    ub, pb = _axis(back_pts) if len(back_pts) >= 2 else (u, np.array(back_pts[0]))

    def back_at(t):
        return pb + ub * (t - float(np.dot(pb - p0, u)))

    centres = sorted(t_of([a["x"], a["y"]]) for a in anchors) if len(anchors) >= 2 else \
        sorted(t_of(np.array(b["contact"]) - depth) for b in bikes)
    pitch = stands.pitch_estimate(centres) if len(centres) >= 2 else 1.1 * float(np.median([b["w"] for b in bikes]))
    if pitch <= 1.0:
        pitch = med_h * 0.5
    centres = [centres[0] - k * pitch for k in range(n_left, 0, -1)] + centres + \
              [centres[-1] + k * pitch for k in range(1, n_right + 1)]
    t_left, t_right = centres[0] - 0.5 * pitch, centres[-1] + 0.5 * pitch
    implied = int(round((t_right - t_left) / pitch))
    if n > implied:
        extra = n - implied
        t_left -= (extra // 2) * pitch
        t_right += (extra - extra // 2) * pitch
        note += f", widened by {extra} slot(s)"

    def clamp(p):
        return [float(min(w - 1, max(0.0, p[0]))), float(min(h - 1, max(0.0, p[1])))]

    def corner_pair(t):
        b = back_at(t) - v * BACK_PAD_FRAC * med_h
        f = b + depth + v * FRONT_PAD_FRAC * med_h
        return clamp(b), clamp(f)

    bl, fl = corner_pair(t_left)
    br, fr = corner_pair(t_right)
    return [fl, fr], [bl, br], note, None


# =========================================================================== fusion

def fuse(camera_id: str, frame_size: tuple[int, int], bikes: list[dict], anchors: list[dict],
         hint: Optional[CalibrationHint], mark_points: list[list[float]], painted: Optional[dict],
         n_frames: int, max_per_frame: int, last_frame_vehicles: int, stand_notes: Optional[str] = None) -> Layout:
    w, h = frame_size
    parking_type = hint.parking_type if hint else "unknown"
    direction = hint.row_direction if hint else "left_to_right"
    notes: list[str] = [f"{n_frames} frame(s), max {max_per_frame} vehicle(s) in one frame"]
    if stand_notes:
        notes.append(stand_notes)

    rows_b = split_rows(bikes)
    rows_a = _assign_anchors_to_rows(anchors, rows_b) if rows_b else ([list(anchors)] if anchors else [])
    rows_b, rows_a, merged = _merge_far_side_rows(rows_b, rows_a)
    if merged:
        notes.append(f"{merged} row(s) of bikes lie right behind a row of stands: treated as its far side")
    if not rows_b and not anchors:
        if hint and hint.corners and hint.slot_count > 0:
            c = [[x * w, y * h] for x, y in hint.corners]
            notes.append(f"no vehicles / stands: row corners and slot count {hint.slot_count} taken from the VLM")
            return _finish(camera_id, frame_size, [(hint.slot_count, [c[3], c[2]], [c[0], c[1]], None)], direction,
                           parking_type, hint, [], notes)
        raise ValueError("cannot propose a layout: no vehicles, no stands and no usable VLM answer - "
                         "park a bike in the row, post a few more frames, or draw the row corners manually")
    if not rows_b:
        rows_b = [[]]
    main = int(np.argmax([len(r) for r in rows_b]))
    if len(rows_b) > 1:
        notes.append(f"{len(rows_b)} rows of bikes found (sizes {[len(r) for r in rows_b]}); VLM answer applied to row {main + 1}")

    prepared = []
    for ri, (rb, ra) in enumerate(zip(rows_b, rows_a)):
        ra = sorted(ra, key=lambda a: a["x"])
        if len(rows_b) > 1 and len(rb) < 2 and len(ra) < 2 and ri != main:
            continue                                   # a lone bike without stands is not a row
        prepared.append((rb, ra, ri == 0))
    main = int(np.argmax([len(rb) for rb, _, _ in prepared]))
    order = [ri for ri in range(len(prepared)) if ri != main] + [main]

    row_specs: dict[int, list[tuple]] = {}
    all_anchor_pts: list[list[float]] = []
    others_total = 0
    for ri in order:
        rb, ra, is_front = prepared[ri]
        is_main = ri == main
        bikes_now = max_per_frame_in_row(rb)
        dividers = painted["back"] if (painted and is_front) else None
        model = row_model(rb, ra, dividers, frame_size)
        if model:
            notes.extend(f"row {ri + 1}: {n}" for n in model.notes)
            pitch = model.lattice.pitch(0.5 * (model.lattice.n_min + model.lattice.n_max))
        else:
            pitch = stands.pitch_estimate([a["x"] for a in ra]) if len(ra) >= 2 else 0.0
            if pitch <= 0 and len(rb) >= 2:
                pitch = stands.pitch_estimate(sorted(b["contact"][0] for b in rb)) or 0.0

        # ---- slot count estimates for this row ------------------------------------------------
        est: dict[str, int] = {}
        if len(rb) >= 1:
            est["bike_clusters"] = _cluster_count([b["contact"][0] for b in rb], float(np.median([b["w"] for b in rb])))
        strong = model is not None and model.n_anchors >= MIN_LATTICE_ANCHORS
        if model:
            est[f"lattice({model.source})"] = model.count
        if is_main and hint:
            vlm_here = hint.slot_count - others_total
            if vlm_here >= bikes_now:
                est["vlm_count"] = vlm_here
                if hint.free_count is not None and len(prepared) == 1:
                    est["last_frame_vehicles_plus_vlm_free"] = last_frame_vehicles + hint.free_count
            if hint.mark_slots and mark_points:
                ok, why = marks_consistent(hint, mark_points, pitch)
                notes.append(f"VLM marks {'accepted' if ok else 'rejected'} ({why})")
        if is_front and painted and painted["spacing_cv"] <= 0.25:
            est["painted_dividers(info)"] = painted["count"]
            if parking_type in ("unknown", "unmarked"):
                parking_type = "painted_bays"
        if strong:
            core = [model.count]
        else:
            core = [v for k, v in est.items() if k.startswith("lattice") or k in ("vlm_count", "last_frame_vehicles_plus_vlm_free")]
        if not core:
            core = [v for k, v in est.items() if k == "bike_clusters"]
        n = int(round(float(np.median(core)))) if core else 1
        n = max(n, bikes_now, 1)
        notes.append(f"row {ri + 1}: estimates " + ", ".join(f"{k}={v}" for k, v in est.items()) + f" -> N={n}")
        if not is_main:
            others_total += n

        # ---- geometry ---------------------------------------------------------------------------
        specs: list[tuple] = []
        if model:
            if n > model.count and model.source == "bikes":
                # more bikes in one frame than lattice slots: the bikes are packed tighter than the fitted
                # pitch -> split their extent equally (dense, full rows)
                ts = sorted(model.t(_mid(b)) for b in rb)
                pitch = (ts[-1] - ts[0]) / max(n - 1, 1)
                if pitch <= 1.0:
                    pitch = model.med_h * 0.3
                old_pitch = model.lattice.pitch(0.5 * (model.lattice.n_min + model.lattice.n_max))
                model.lattice = lattice.affine(ts[0], pitch, 0, n - 1)
                model.bias = 0.0
                model.bike_slots = [int(min(n - 1, max(0, round(model.lattice.n(model.t(_mid(b))))))) for b in rb]
                for dm in model.sides.values():
                    if dm.const is None:
                        dm.rho *= old_pitch / pitch            # keep the slot depth in pixels
                notes.append(f"row {ri + 1}: {n} bikes in one frame but {model.count} lattice slots: extent split equally")
            elif n != model.count:
                model.lattice = _rerange(model.lattice, n, model.bike_slots)
                notes.append(f"row {ri + 1}: lattice range adjusted to N={n}")
            polys_by_side = row_polygons(model, frame_size)
            for sign in sorted(polys_by_side, reverse=True):
                polys = polys_by_side[sign]
                specs.append((n, [polys[0][3], polys[-1][2]], [polys[0][0], polys[-1][1]], polys))
            all_anchor_pts.extend(model.back_at(model.lattice.u(k)).tolist() for k in model.lattice.indices())
        else:
            corners_px = [[x * w, y * h] for x, y in hint.corners] if (is_main and hint and hint.corners) else None
            front, back, gnote, polys = row_geometry(rb, ra, n, frame_size, corners_hint=corners_px)
            notes.append(f"row {ri + 1}: {gnote} (affine fallback)")
            specs.append((n, front, back, polys))
            all_anchor_pts.extend([[a["x"], a["y"]] for a in ra])
        row_specs[ri] = specs
    flat = [spec for ri in range(len(prepared)) for spec in row_specs[ri]]
    if parking_type == "unknown" and any(len(row_specs[ri]) == 2 for ri in row_specs):
        parking_type = "stirrup_rack"

    return _finish(camera_id, frame_size, flat, direction, parking_type, hint, all_anchor_pts, notes,
                   mark_points=mark_points)


def _cluster_count(xs: list[float], med_w: float) -> int:
    xs = sorted(xs)
    n = 1
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a > config.CLUSTER_GAP_FRAC * med_w:
            n += 1
    return n


def _finish(camera_id, frame_size, row_specs, direction, parking_type, hint, anchor_pts, notes,
            mark_points=None) -> Layout:
    rows = [Row(row_id=f"r{i + 1}", slot_count=int(n), front=normalize(front, frame_size),
                back=normalize(back, frame_size), direction=direction)
            for i, (n, front, back, _) in enumerate(row_specs)]
    occupancy_mode = "count_only" if parking_type == "dense_rack" else "per_slot"
    layout = Layout(camera_id=camera_id, frame_size=list(frame_size), parking_type=parking_type, status="proposed",
                    source="auto", occupancy_mode=occupancy_mode, rows=rows, slots=[],
                    anchors=normalize(anchor_pts, frame_size), updated_at=utcnow())
    layout.slots = slots_from_rows(layout)
    # rows with a lattice get their explicit (perspective, unevenly spaced) polygons instead of the equal split
    for row, (n, _, _, polys) in zip(rows, row_specs):
        if polys and len(polys) == n:
            row_slots = [s for s in layout.slots if s.row_id == row.row_id]
            if direction == "right_to_left":
                polys = list(reversed(polys))
            for s, poly in zip(row_slots, polys):
                s.polygon = normalize(poly, frame_size)

    if hint and hint.mark_slots and mark_points:
        agree = total = 0
        for mark, j in hint.mark_slots.items():
            if 1 <= mark <= len(mark_points) and 1 <= j <= len(layout.slots):
                total += 1
                poly = pixel_polygon(layout.slots[j - 1].polygon, frame_size)
                agree += point_in_polygon(poly, mark_points[mark - 1])
        if total:
            notes.append(f"VLM mark check: {agree}/{total} marked bikes fall in the slot the VLM named")
    if hint:
        notes.append(hint.notes)
    layout.calibration_hint = CalibrationHint(
        source=hint.source if hint else "local", parking_type=parking_type, slot_count=len(layout.slots),
        row_direction=direction, corners=hint.corners if hint else None,
        free_count=hint.free_count if hint else None, occupied_count=hint.occupied_count if hint else None,
        mark_slots=hint.mark_slots if hint else None, stand_count=len(anchor_pts) or None,
        notes=" | ".join(notes))
    return layout
