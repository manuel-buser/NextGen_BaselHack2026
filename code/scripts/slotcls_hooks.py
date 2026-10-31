"""Find the hook row of a wall rack in a roughly frontal picture and write its slot layout (N equal slots).

For phone pictures of the Memox wall rack (and a camera mounted head-on): the two pillars are the dark columns
that reach from above the rail to the floor, the rail is the dark bar between them, the hooks hang below it at a
constant pitch. This finds the pillars, the rail and the hook lattice, checks that N hooks fit between the pillars
and writes ``<image>.layout.json`` plus ``<image>.overlay.jpg`` and ``.crops.jpg``, the same files
slotcls_calibrate.py writes from clicked corners. For the angled ESP32 view use that script.

    .venv/Scripts/python code/scripts/slotcls_hooks.py --images "assets/pictures/DemoPictures/memox_*.jpeg" --slots 19

Prints one line per picture: pillars, rail rows, pitch, how many hook stems lie on the lattice, the slot span, and
"uniform" when no lattice fitted and the span between the pillars was split evenly (check the overlay then).
Overlay marks: magenta circles = the N hook positions used, orange = dark stems found under the rail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bikeslot.geometry import imread  # noqa: E402
from bikeslot.slotcls.crops import contact_sheet, slot_crops  # noqa: E402
from slotcls_calibrate import draw_overlay, layout_from_corners  # noqa: E402

RAIL_GREY = 70      # the rail, the pillars and the skirting are almost black (grey 25-55); the wall is 180-220;
                    # a brighter exposure lifts the rail to ~100, so rail_grey() scales this with the wall
STEM_GREY = 120     # hook stems below the rail, a little lighter at their edges
ROW_DEPTH = 3.3     # slot bottom = rail bottom + this many pitches (wall foot plus a little floor)


def runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """[start, end] index pairs of True runs, bridging gaps up to max_gap."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


GREY = RAIL_GREY    # threshold in use: the fixed one first, the adaptive one when that finds no lattice


def rail_grey(gray: np.ndarray) -> int:
    return GREY


def adaptive_grey(gray: np.ndarray) -> int:
    """Darkness threshold scaled with the wall brightness (half of it, within sensible bounds)."""
    h, w = gray.shape
    wall = float(np.median(gray[int(0.25 * h):int(0.75 * h), int(0.3 * w):int(0.7 * w)]))
    return int(min(110, max(55, 0.5 * wall)))


def find_rail(gray: np.ndarray) -> tuple[int, int]:
    """Top and bottom row of the rail: the topmost thin dark band that spans the middle of the picture
    (the skirting and the floor below are dark too, but lower and thicker)."""
    h, w = gray.shape
    y0, y1 = int(0.25 * h), int(0.75 * h)
    band = gray[y0:y1, int(0.3 * w):int(0.7 * w)] < rail_grey(gray)
    share = band.mean(axis=1)
    groups = [g for g in runs(share > 0.45, max_gap=2) if 4 <= g[1] - g[0] <= 30]
    if not groups:
        raise ValueError("no rail found (no thin dark band spanning the middle of the picture)")
    top, bottom = groups[0]
    return y0 + top, y0 + bottom


def find_pillars(gray: np.ndarray, rail_top: int) -> list[tuple[int, int]]:
    """Column runs that are dark over most of the wall above the rail: the pillars (their bright specular stripe
    is bridged). Graffiti strokes are thin and do not fill the band."""
    h, w = gray.shape
    y0, y1 = int(0.3 * h), max(int(0.3 * h) + 20, rail_top - 25)
    share = (gray[y0:y1, :] < rail_grey(gray) + 30).mean(axis=0)
    return [r for r in runs(share > 0.4, max_gap=45) if r[1] - r[0] >= 20 or r[0] == 0 or r[1] == w - 1]


def rail_extent(gray: np.ndarray, top: int, bottom: int) -> tuple[int, int]:
    """Leftmost and rightmost column of the rail: the longest dark run along its rows (a few px of tilt and
    gaps of a few px allowed). The pillars are as dark as the rail, so this can run into them."""
    rows = gray[max(0, top - 8):bottom + 9] < rail_grey(gray)
    share = rows.mean(axis=0)
    spans = runs(share > 0.3, max_gap=12)
    left, right = max(spans, key=lambda s: s[1] - s[0])
    return left, right


def wall_span(gray: np.ndarray, rail_top: int, rail_bottom: int) -> tuple[int, int, str]:
    """Inner edges of the two pillars around the image centre. Without a pillar on a side: the end of the rail
    when it ends inside the picture, else the picture's edge. Returns (left, right, how)."""
    h, w = gray.shape
    centre = w // 2
    pillars = find_pillars(gray, rail_top)
    rail_left, rail_right = rail_extent(gray, rail_top, rail_bottom)
    left_p = [p for p in pillars if p[1] < centre]
    right_p = [p for p in pillars if p[0] > centre]
    if left_p:
        left, how_l = left_p[-1][1] + 1, "pillar"
    elif rail_left > 15:
        left, how_l = rail_left, "rail end"
    else:
        left, how_l = 0, "edge"
    if right_p:
        right, how_r = right_p[0][0] - 1, "pillar"
    elif rail_right < w - 16:
        right, how_r = rail_right, "rail end"
    else:
        right, how_r = w - 1, "edge"
    return left, right, f"{how_l}/{how_r}"


def rail_top_at(gray: np.ndarray, x0: int, x1: int, top: int, bottom: int) -> float:
    """The rail's top row within the columns x0..x1 (hand-held pictures are a little tilted)."""
    lo = max(0, top - 15)
    window = gray[lo:bottom + 15, x0:x1] < rail_grey(gray)
    share = window.mean(axis=1)
    groups = runs(share > 0.45, max_gap=2)
    return float(lo + groups[0][0]) if groups else float(top)


def hook_candidates(gray: np.ndarray, rail_bottom: int, left: int, right: int) -> list[float]:
    """x positions of narrow dark stems just below the rail: hooks, but also bike tubes (the lattice sorts them)."""
    band = gray[rail_bottom + 6:rail_bottom + 40, :] < rail_grey(gray) + 50
    dark = np.convolve(band.mean(axis=0), np.ones(3) / 3, mode="same")
    out = []
    for start, end in runs(dark > 0.35, max_gap=1):
        width = end - start + 1
        if start >= left and end <= right and 2 <= width <= 28:
            seg = dark[start:end + 1]
            out.append(start + float(np.average(np.arange(len(seg)), weights=seg)))
    return out


def fit_lattice(xs: list[float], pitch_lo: float, pitch_hi: float) -> tuple[float, float, int] | None:
    """Regular lattice x0 + k*p (pitch within the given range) through the most candidates."""
    if len(xs) < 4:
        return None
    xs = np.array(xs)
    best = None
    for p in np.arange(pitch_lo, pitch_hi, 0.25):
        tol = 0.09 * p
        for a in xs:
            k = np.round((xs - a) / p)
            inl = np.abs(xs - (a + k * p)) < tol
            score = len(set(k[inl]))
            if best is None or score > best[0]:
                best = (score, p, a, inl)
    score, p, a, inl = best
    if score < 5:
        return None
    for _ in range(2):  # least-squares refinement on the inliers
        k = np.round((xs - a) / p)
        A = np.stack([np.ones_like(xs[inl]), k[inl]], axis=1)
        a, p = np.linalg.lstsq(A, xs[inl], rcond=None)[0]
        inl = np.abs(xs - (a + np.round((xs - a) / p) * p)) < 0.09 * p
    return float(a), float(p), int(inl.sum())


def hook_row(img: np.ndarray, n_slots: int) -> dict:
    """Try the fixed darkness threshold first, then the adaptive one; the first attempt with a lattice wins."""
    global GREY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    attempts, error = [], None
    for GREY in (RAIL_GREY, adaptive_grey(gray)):
        try:
            found = _hook_row(gray, n_slots)
        except ValueError as exc:
            error = exc
            continue
        found["grey"] = GREY
        # plausibility: a rail is 8-20 rows thick (the skirting is thicker) and most hooks must sit on the lattice
        found["plausible"] = found["rail"][1] - found["rail"][0] <= 20 and found["inliers"] >= 7
        if not found["uniform"] and found["plausible"]:
            return found
        attempts.append(found)
    if attempts:
        return attempts[0]
    raise error


def _hook_row(gray: np.ndarray, n_slots: int) -> dict:
    h, w = gray.shape
    top, bottom = find_rail(gray)
    left, right, how = wall_span(gray, top, bottom)
    span = right - left
    third = max(20, span // 3)
    y_left = rail_top_at(gray, left, left + third, top, bottom)
    y_right = rail_top_at(gray, right - third, right, top, bottom)
    cands = hook_candidates(gray, bottom, left, right)
    # N hooks between the pillars: the pitch is close to span / N (the end hooks sit half a pitch from the pillars)
    fit = fit_lattice(cands, span / (n_slots + 1.6), span / (n_slots - 0.6))
    if fit is not None:
        x0, p, inliers = fit
        ideal_first = left + (span - (n_slots - 1) * p) / 2
        k0 = round((ideal_first - x0) / p)
        hooks = [x0 + (k0 + i) * p for i in range(n_slots)]
        uniform = False
    else:
        p = span / n_slots
        hooks = [left + p * (i + 0.5) for i in range(n_slots)]
        inliers, uniform = 0, True
    x_first, x_last = hooks[0] - p / 2, hooks[-1] + p / 2
    y_bottom = min(h - 1, bottom + ROW_DEPTH * p)
    corners = [[x_first, y_left], [x_last, y_right], [x_last, y_bottom], [x_first, y_bottom]]
    return {"corners": corners, "rail": (top, bottom), "span": (left, right), "span_how": how, "pitch": p,
            "hooks": hooks, "candidates": cands, "inliers": inliers, "uniform": uniform}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--slots", type=int, default=19)
    ap.add_argument("--site", default="site01")
    ap.add_argument("--site-name", default="Memox wall rack")
    ap.add_argument("--camera", help="camera id for all pictures (default: the picture's file stem)")
    args = ap.parse_args()
    paths = []
    for spec in args.images:
        p = Path(spec)
        paths += sorted(p.parent.glob(p.name)) if any(ch in spec for ch in "*?[") else [p]
    for path in paths:
        img = imread(str(path))
        if img is None:
            print(f"{path.name}: cannot read")
            continue
        try:
            found = hook_row(img, args.slots)
        except ValueError as exc:
            print(f"{path.name}: {exc}")
            continue
        camera = args.camera or path.stem
        layout = layout_from_corners(camera, (img.shape[1], img.shape[0]), found["corners"], args.slots,
                                     site_id=args.site, site_name=args.site_name)
        out = path.with_suffix(".layout.json")
        out.write_text(layout.model_dump_json(indent=2), encoding="utf-8")
        overlay = draw_overlay(img, layout)
        for x in found["candidates"]:
            cv2.circle(overlay, (int(round(x)), found["rail"][1] + 50), 5, (0, 165, 255), 2)
        for x in found["hooks"]:
            cv2.circle(overlay, (int(round(x)), found["rail"][1] + 22), 6, (255, 0, 255), 2)
        cv2.imwrite(str(path.with_suffix(".overlay.jpg")), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crops = slot_crops(layout, img)
        cv2.imwrite(str(path.with_suffix(".crops.jpg")), contact_sheet(crops), [cv2.IMWRITE_JPEG_QUALITY, 85])
        c = found["corners"]
        print(f"{path.name}: grey<{found['grey']}, wall x {found['span'][0]}-{found['span'][1]} ({found['span_how']}), rail y {found['rail'][0]}-{found['rail'][1]}, "
              f"pitch {found['pitch']:.1f} px, {found['inliers']} of {len(found['candidates'])} stems on the lattice, "
              f"slots x {c[0][0]:.0f}-{c[1][0]:.0f}, bottom y {c[2][1]:.0f}"
              f"{'  UNIFORM SPLIT - check the overlay' if found['uniform'] else ''}"
              f"{'  IMPLAUSIBLE - check the overlay' if not found.get('plausible', True) else ''}")


if __name__ == "__main__":
    main()
