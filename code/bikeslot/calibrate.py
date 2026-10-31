"""Stage A: propose a slot layout for a camera from its buffered frames.

Evidence, from the most to the least reliable:
  * rack stands (stands.py): one fixed anchor per slot -> the row AXIS and the back edge of the slots;
    stands hidden behind bikes are inferred from the spacing;
  * parked bikes (ground-contact points of the masks) -> the front edge of the slots (depth vector from
    stand base to wheel contact), a lower bound on the count, and the split into rows;
  * the VLM (vlm.py): parking type, enumerated slot count, numbering direction, rough row corners, and
    "bike k stands in slot j" (used only when consistent with the bike spacing);
  * painted dividers (lines.py): an extra count estimate only.
Slot count = median of the estimates, never below the number of bikes seen in one frame.  Each row is a
quadrilateral (back line along the stands, front line = back line + depth vector) split equally on the
ground plane.  Several rows are supported (bikes are clustered by contact height); the VLM answer applies
to the row with the most bikes.  Result: Layout(status="proposed") for the admin to confirm.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config, detector, stands, vlm
from .geometry import normalize, pixel_polygon, point_in_polygon, slots_from_rows
from .preview import blur_boxes
from .schemas import CalibrationHint, Detection, Layout, Row, utcnow

FRONT_PAD_FRAC = 0.08      # front edge pushed towards the camera by this share of the bike height
BACK_PAD_FRAC = 0.05       # back edge pushed behind the stand bases by this share of the bike height
MIN_CONF_FOR_LAYOUT = 0.3  # detections below this are ignored for calibration
ROW_SPLIT_FRAC = 0.45      # contact points further apart (across the row) than this share of the bike height = new row


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
    """Contact point (front, on the ground), far end (handlebar area) and size of one detected bike."""
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


def end_slots_from_bikes(bikes: list[dict], anchors: list[dict]) -> tuple[int, int]:
    """Bikes parked clearly beyond the outermost stands (along the row axis) imply extra end slots."""
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


def row_geometry(bikes: list[dict], anchors: list[dict], n: int, frame_size: tuple[int, int],
                 n_left: int = 0, n_right: int = 0, corners_hint=None) -> tuple[list, list, str, Optional[list]]:
    """Build the row quadrilateral.  Returns (front [P0,P1], back [Q0,Q1], note, polygons) in pixels;
    ``polygons`` is a list of n slot quads with boundaries at the stand midpoints when the stands account
    for every slot, else None (caller splits the row equally)."""
    w, h = frame_size
    contacts = np.array([b["contact"] for b in bikes], dtype=np.float64) if bikes else np.zeros((0, 2))
    med_h = float(np.median([b["h"] for b in bikes])) if bikes else (float(np.median([a["h"] for a in anchors])) * 3.0)

    # axis along the stands (best) or along the bikes' contact points
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
        v = -v                                   # depth points towards the camera (down in the image)

    def t_of(p):
        return float(np.dot(np.asarray(p) - p0, u))

    def along(t):
        return p0 + u * t

    # depth vector: stand base -> wheel contact (matched by nearest stand along the axis), else bike lean
    depths = []
    if anchors and len(contacts):
        at = np.array([t_of([a["x"], a["y"]]) for a in anchors])
        pitch = stands.pitch_estimate(at.tolist()) if len(at) >= 2 else med_h
        for b in bikes:
            tb = t_of(b["far"])
            k = int(np.argmin(np.abs(at - tb)))
            if abs(at[k] - tb) <= 0.8 * max(pitch, 1.0):
                a = anchors[k]
                depths.append((tb, np.array(b["contact"]) - np.array([a["x"], a["y"]])))
    if not depths and bikes:
        depths = [(t_of(b["far"]), np.array(b["contact"]) - np.array(b["far"])) for b in bikes]
    if depths:
        ts = np.array([t for t, _ in depths])
        ds = np.array([d for _, d in depths])
        if len(depths) >= 3 and np.ptp(ts) > 2.0 * med_h:
            cx = np.polyfit(ts, ds[:, 0], 1)
            cy = np.polyfit(ts, ds[:, 1], 1)

            def depth(t):
                return np.array([np.polyval(cx, t), np.polyval(cy, t)])
        else:
            dmed = np.median(ds, axis=0)

            def depth(t):
                return dmed
    else:
        stand_h = float(np.median([a["h"] for a in anchors]))

        def depth(t):
            return v * config.SLOT_DEPTH_PER_STAND_HEIGHT * stand_h

    # back line: through the stand bases, else through contact - depth
    if len(anchors) >= 2:
        back_pts = [[a["x"], a["y"]] for a in anchors]
    else:
        back_pts = [(np.array(b["contact"]) - depth(t_of(b["far"]))).tolist() for b in bikes]
    ub, pb = _axis(back_pts) if len(back_pts) >= 2 else (u, np.array(back_pts[0]))

    def back_at(t):
        return pb + ub * (t - float(np.dot(pb - p0, u)))

    # extent along the axis from the slot centres we know (stands first, else bikes) + end slots
    if len(anchors) >= 2:
        centres = sorted(t_of([a["x"], a["y"]]) for a in anchors)
    else:
        centres = sorted(t_of(np.array(b["contact"]) - depth(t_of(b["far"]))) for b in bikes)
    pitch = stands.pitch_estimate(centres) if len(centres) >= 2 else 1.1 * float(np.median([b["w"] for b in bikes]))
    if pitch <= 1.0:
        pitch = med_h * 0.5
    # slot centres incl. the end slots implied by bikes / the VLM
    centres = [centres[0] - k * pitch for k in range(n_left, 0, -1)] + centres + \
              [centres[-1] + k * pitch for k in range(1, n_right + 1)]
    t_left = centres[0] - 0.5 * pitch
    t_right = centres[-1] + 0.5 * pitch
    implied = int(round((t_right - t_left) / pitch))
    if n > implied:                               # more slots than the extent holds: widen symmetrically
        extra = n - implied
        t_left -= (extra // 2) * pitch
        t_right += (extra - extra // 2) * pitch
        note += f", widened by {extra} slot(s)"

    def clamp(p):
        return [float(min(w - 1, max(0.0, p[0]))), float(min(h - 1, max(0.0, p[1])))]

    def corner_pair(t):
        b = back_at(t) - v * BACK_PAD_FRAC * med_h
        f = b + depth(t) + v * FRONT_PAD_FRAC * med_h
        return clamp(b), clamp(f)

    bl, fl = corner_pair(t_left)
    br, fr = corner_pair(t_right)

    polygons = None
    if len(anchors) >= 3 and len(centres) == n:
        # one stand per slot: put the boundaries halfway between neighbouring stands (handles uneven spacing)
        bounds = [t_left] + [(a + b) / 2.0 for a, b in zip(centres[:-1], centres[1:])] + [t_right]
        polygons = []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            b0, f0 = corner_pair(t0)
            b1, f1 = corner_pair(t1)
            polygons.append([b0, b1, f1, f0])
        note += ", slot boundaries at stand midpoints"
    return [fl, fr], [bl, br], note, polygons


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

    # the VLM describes the whole scene; its count is applied to the main row minus what the other rows hold
    prepared = []
    for ri, (rb, ra) in enumerate(zip(rows_b, rows_a)):
        ra = sorted(ra, key=lambda a: a["x"])
        if len(ra) >= 3:
            ra = stands._fill_gaps(ra)
        if len(rows_b) > 1 and len(rb) < 2 and len(ra) < 2 and ri != main:
            continue                                   # a lone bike without stands is not a row
        prepared.append((rb, ra))
    main = int(np.argmax([len(rb) for rb, _ in prepared]))
    order = [ri for ri in range(len(prepared)) if ri != main] + [main]

    row_specs: dict[int, tuple] = {}
    all_anchor_pts = []
    others_total = 0
    for ri in order:
        rb, ra = prepared[ri]
        is_main = ri == main
        bikes_now = max_per_frame_in_row(rb)
        pitch = stands.pitch_estimate([a["x"] for a in ra]) if len(ra) >= 2 else 0.0
        if pitch <= 0 and len(rb) >= 2:
            pitch = stands.pitch_estimate(sorted(b["contact"][0] for b in rb)) or 0.0

        # ---- slot count estimates for this row ------------------------------------------------
        est: dict[str, int] = {}
        if len(rb) >= 1:
            est["bike_clusters"] = _cluster_count([b["contact"][0] for b in rb], float(np.median([b["w"] for b in rb])))
        n_left, n_right = end_slots_from_bikes(rb, ra) if len(ra) >= 3 else (0, 0)
        if is_main and hint:
            vlm_here = hint.slot_count - others_total
            if vlm_here >= bikes_now:
                est["vlm_count"] = vlm_here
                if hint.free_count is not None and len(prepared) == 1:
                    est["last_frame_vehicles_plus_vlm_free"] = last_frame_vehicles + hint.free_count
            if hint.mark_slots and mark_points:
                ok, why = marks_consistent(hint, mark_points, pitch)
                notes.append(f"VLM marks {'accepted' if ok else 'rejected'} ({why})")
                if ok and ra:
                    # how many slots lie left of the first detected stand? the smallest slot index given to a
                    # bike whose contact lies left of the first stand tells us
                    first_x = ra[0]["x"]
                    left_idx = [j for m, j in hint.mark_slots.items()
                                if 1 <= m <= len(mark_points) and mark_points[m - 1][0] < first_x - 0.25 * pitch and j >= 1]
                    if left_idx:
                        n_left = max(n_left, max(left_idx))
        if is_main and painted and painted["spacing_cv"] <= 0.25:
            est["painted_dividers(info)"] = painted["count"]
            if parking_type in ("unknown", "unmarked"):
                parking_type = "painted_bays"
        if len(ra) >= 3:
            est["stands"] = len(ra) + n_left + n_right
        core = [v for k, v in est.items() if k in ("stands", "vlm_count", "last_frame_vehicles_plus_vlm_free")]
        if not core:
            core = [v for k, v in est.items() if k == "bike_clusters"]
        n = int(round(float(np.median(core)))) if core else 1
        n = max(n, bikes_now, 1)
        notes.append(f"row {ri + 1}: estimates " + ", ".join(f"{k}={v}" for k, v in est.items()) + f" -> N={n}")
        if not is_main:
            others_total += n

        corners_px = [[x * w, y * h] for x, y in hint.corners] if (is_main and hint and hint.corners) else None
        front, back, gnote, polys = row_geometry(rb, ra, n, frame_size, n_left, n_right, corners_hint=corners_px)
        notes.append(f"row {ri + 1}: {gnote}")
        row_specs[ri] = (n, front, back, polys)
        all_anchor_pts.extend([[a["x"], a["y"]] for a in ra])
    row_specs = [row_specs[ri] for ri in range(len(prepared))]

    return _finish(camera_id, frame_size, row_specs, direction, parking_type, hint, all_anchor_pts, notes,
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
    # rows with one stand per slot get their explicit (unevenly spaced) polygons instead of the equal split
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
