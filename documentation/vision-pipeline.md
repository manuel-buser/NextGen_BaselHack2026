# Vision pipeline and API contract (`code/bikeslot`)

Turns camera snapshots of a bicycle parking row into **which slots exist, how they are numbered and which
are free**, and serves that as JSON for the frontend.

## Overview

```
ESP32-CAM ──JPEG every ~5 s──▶ POST /cameras/{id}/frames ──▶ Stage B: occupancy ──▶ CameraStatus JSON
                                                                     ▲
   admin: POST /cameras/{id}/calibrate ──▶ Stage A: layout proposal ─┘  (admin confirms via PUT .../layout)
   frontend: GET /sites, GET /cameras/{id}/status, GET /cameras/{id}/preview.jpg
```

> **Second approach (2026-10-31 evening): fixed slot polygons + per-slot binary classifier.** For a camera that
> never moves, the slots never move either, so nothing has to find or count them per frame: polygons are drawn
> once, every snapshot is cut into one crop per slot and a small CNN says free/occupied. It lives next to this
> pipeline in `code/bikeslot/slotcls/` and is selected per camera (`layout.params.engine = "classifier"`) or
> globally (`BIKESLOT_ENGINE=classifier`); the API below is unchanged. See
> [slot-classifier.md](slot-classifier.md). The detector pipeline described here stays the default.

Two stages per camera:

| Stage | When | What |
|---|---|---|
| **A – calibration** (`calibrate.py`, `lattice.py`, `stands.py`, `vlm.py`, `lines.py`) | once per camera, admin-triggered | Proposes the slot layout: how many slots, in which order, where in the image. Evidence: **rack stands** found by YOLOE visual prompting (one example stand found by text prompt, then all look-alikes), **parked bikes** (mask contact point and centroid: row split, slot depth, which side of the rack is used, extra end slots), **painted dividers** and a **GPT-vision set-of-mark answer** (parking type, enumerated slots). Per row a **1-D projective lattice** (equally spaced slots seen in perspective) is fitted to the stands, dividers and bikes: it places the stands hidden behind bikes, puts the slot boundaries at perspective-correct positions and decides whether an end bike occupies an extra slot or just leans. Several rows and racks used from both sides are supported. The admin then confirms or corrects the result. |
| **B – occupancy** (`detector.py`, `occupancy.py`) | every snapshot, fully local | YOLO11x-seg finds bicycles / motorcycles / persons. Each vehicle's **ground-contact point** (lowest pixels of its mask ≈ rear wheel) and its footprint are matched against the slot polygons. One vehicle fills at most one slot. Votes are smoothed over the last 5 frames with hysteresis. |

### Why this design
* **Local detector + geometry per frame**: ~100 ms on the laptop GPU, offline, deterministic, explainable.
  The mask contact point (not the box centre) is what makes bikes leaning over a line land in the right bay.
* **VLM only at calibration**: good at "what kind of parking, how many slots, which direction", bad at
  pixel-precise coordinates, too slow/expensive per frame and needs internet.
* **Row model**: a layout is a front line + back line + slot count. The admin drags 4 corners and sets N
  instead of drawing N polygons. Slots are split equally *on the ground* (homography), so perspective is
  respected. Explicit polygons are stored too and can be edited individually.
* **Projective lattice instead of a pixel pitch** (`lattice.py`): along a row seen in perspective the
  stand spacing is not constant, it shrinks towards the horizon as `u(n) = (a·n + b) / (c·n + 1)`.
  Comparing a gap at the far end with the median pitch confuses perspective with "a stand is hidden here"
  (that is where the old 8-or-11-instead-of-9 came from). Fitting the three parameters with an
  integer-gap RANSAC makes every slot exactly 1.0 wide in lattice units: hidden stands sit at the missing
  integers, bikes leaning out of the last slot stay below index N, bikes in a hidden end slot land at
  index N. Stands, painted dividers and bikes each propose a lattice; the one that explains most of the
  evidence wins (on the golden scene that is the painted bays - its stands are drawn off-centre).
* **Snapshots instead of a video stream**: parking state changes slowly, stills have more pixels, less WiFi
  load, fewer frames ever touch the laptop.
* **count_only mode** for dense double racks where per-slot assignment is meaningless: reports an estimated
  free count instead of pretending.

## Results on the golden data set (`assets/pictures/GoldenDataSet`)

| Check | Result |
|---|---|
| Per-slot occupancy with the golden layout (3 images × 9 slots) | **27 / 27** |
| Detector latency, 1280 px, RTX 4070 laptop, fp16 | ~80 ms detect, ~100 ms whole frame |
| Auto layout, row model with the true corners and N | mean IoU 0.85 vs hand-labelled bays |
| Auto layout straight from calibration, 3 frames, no VLM, no admin | **9 slots, 27/27**, mean IoU 0.74 vs the hand-labelled bays |

### Automatic calibration benchmark (2026-10-31, `code/scripts/calib_bench.py --vlm 0`)

Occupancy = per-slot states of the proposal applied to the same images, *without any admin correction*.
Golden truth is 9 slots; the ChatGPT images have no labels, so "placed" = bikes that got a slot of their
own (slots occupied / bikes detected). "Before" is the previous pixel-pitch heuristic (same day, same images).

| Scene | Stands found | Proposed N (before → now) | Raw proposal (before → now) |
|---|---|---|---|
| golden, 3 frames with different occupancy | 7 of 10 drawn | 11 → **9** | 17/27 → **27/27** |
| golden, single frame 001 / 002 / 003 | 5 | 10 / - / 8 → 9 / 9 / 10 | 16 / - / 11 of 27 → 17 / 23 / 16 of 27 |
| gpt_01 (2 rows, painted bays, overhead) | 8 | 19 → **23** (= 11 + 12 bays) | 14/15 → **15/15** |
| gpt_02 (2 dense diagonal rows) | none | 22 → 23 | 11/25 → 16/25 |
| gpt_03 (2 full rows) | none | 19 → 19 | 19/19 → 19/19 |
| gpt_04 (stirrups, diagonal, one bike behind the rack) | 5 | 11 → 12 | 5/8 → 7/8 |
| gpt_05 (2 rows of wheel holders, a few empty) | 4 | 18 → 22 | 14/20 → **19/20** |
| gpt_06 (2 dense rows, 34 bikes) | none | 31 → 31 | 24/34 → 29/34 |
| gpt_07 (2 diagonal rows, painted bays) | 5 | 19 → 19 (= 8 + 11 bays) | 1/14 → **14/14** |

What this means in practice:
* **The golden scene calibrates itself** with three frames of different occupancy: the painted bays form
  the lattice, the bikes extend it to the wall, the two stirrups-in-one-bay quirks are outliers. The polygons'
  back edge runs through the far ends of the bikes (slightly above the painted back line), which is why the
  IoU is 0.74 and not higher; the occupancy is unaffected.
* **Calibrate after a few frames with different occupancy.** A single golden frame still gives 9-10 slots
  with the right pitch but can be one slot off at an end; the admin fixes that with one click.
* **Rows with stands or painted bays** (gpt_01, gpt_05, gpt_07) now get the empty slots between the bikes
  right, including the far rows of diagonal scenes.
* **Where it still needs the admin**: very dense double racks (gpt_02, gpt_06: bikes overlap, the detector
  merges some, per-slot assignment is a guess - `count_only` mode is the honest answer there), and
  hand-drawn irregular racks (gpt_04's stirrups are not evenly spaced in the AI image: the count is right,
  two bikes share a slot).
* GPT-4.1's "bike k is in slot j" answers are wrong more often than right on these scenes (it tends to
  number the bikes 1..k); the consistency check rejects them in that case so they cannot do harm. The
  lattice count overrides the VLM count whenever at least 3 stands or 4 dividers lie on the lattice.
* Tried and dropped: a second YOLOE pass with 2-3 exemplar stands as a multi-box visual prompt found
  exactly the same stands on every scene; the inferred positions from the lattice are what matter.

Flags: `POST /cameras/{id}/calibrate?use_stands=true&use_vlm=true`. The stand model (`yoloe-11l-seg.pt`,
downloaded on first use together with its text encoder) is warmed up in the background at service start;
the first calibration can take ~30 s if it is not ready yet, later ones take 1–5 s.

## Running

```bash
# once
.venv/Scripts/python -m pip install -r requirements.txt
# .env at the repo root with  OPENAI_API_KEY=...   (or MY_API_KEY=...)   - optional, calibration only

# service (the golden layout is loaded for cam01 automatically)
.venv/Scripts/python -m uvicorn --app-dir code bikeslot.api:app --host 0.0.0.0 --port 8000
# docs: http://127.0.0.1:8000/docs   debug view: http://127.0.0.1:8000/
# showcase: http://127.0.0.1:8000/demo  (live camera + annotated golden and auto-calibrated examples)

# live camera: the service polls http://192.168.21.95/snapshot itself as camera live01 (pull mode).
# Change/disable with BIKESLOT_PULL_CAMERAS="live01=http://<ip>/snapshot,cam02=http://<ip2>/snapshot" (or "")
# and BIKESLOT_PULL_INTERVAL=5, or at runtime: PUT /cameras/live01/source?url=http://<ip>/snapshot
# pull state (frames, errors, last fetch) is in GET /health under "pull".

# fake camera: loop the golden images into cam01 every 3 s
.venv/Scripts/python code/scripts/replay.py --camera cam01 --source assets/pictures/GoldenDataSet --interval 3 --loop

# evaluation and tests
.venv/Scripts/python code/scripts/eval_golden.py --previews data/preview/eval
.venv/Scripts/python code/scripts/calib_bench.py --vlm 0        # automatic calibration on all scenes (table above)
cd code && ../.venv/Scripts/python -m pytest tests -q
```

`code` is also the name of a Python standard-library module, so the package is always run with
`--app-dir code` / `PYTHONPATH=code` and never imported as `code.bikeslot`.
Use `127.0.0.1`, not `localhost`, in clients on Windows (IPv6 fallback costs ~2 s per connection).

Tunables live in `code/bikeslot/config.py` and can be overridden with `BIKESLOT_*` environment variables
(e.g. `BIKESLOT_MODEL=yolo11x.pt` for the detection-only model, `BIKESLOT_CONF=0.15`, `BIKESLOT_DEVICE=cpu`).

## Handing the result to the app (`code/app`)

The app shows per-rack counts on its user and admin pages. A rack configured with `vision: {url, camera}` in the
app's racks file takes this service's decided slots instead of pictures (`code/app/README.md`, "Live rack from the
vision service"; contract in `documentation/CONTRACTS.md`, section 2b):

* **Pull** (default): the app calls `GET /cameras/{camera}/status` every few seconds and `GET
  /cameras/{camera}/preview.jpg` for its admin page. Nothing to configure here beyond the firewall (port 8000).
* **Push** (`forward.py`): for an app this laptop can reach but that cannot reach the laptop (a server), set
  `BIKESLOT_APP_URL=http://<app>` and `BIKESLOT_APP_RACKS=live01=MX-CAM` (camera id here -> rack id there, optionally
  `BIKESLOT_APP_TOKEN` = the app's `VISION_TOKEN`). Every new status is posted to `POST {app}/api/vision/{rack}/status`
  from a background thread, newest status per camera only; the app being down never delays a frame. State in
  `GET /health` under `forward` (sent, errors, the app's last reply with its free count).

The live camera `live01` is seeded at start with the 19-hook layout of the demo picture and the classifier engine
(`BIKESLOT_DEMO_LAYOUT_CAMERAS`, default `live01`; a layout saved under `data/layouts/live01.json` wins), so a
camera mounted where the demo picture was taken reports the 19 hooks immediately. Set-up steps for the pitch:
`documentation/demo-runbook.md`.

## API (FastAPI, interactive docs at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/cameras/{camera_id}/frames` | Ingest one snapshot: raw JPEG body (`Content-Type: image/jpeg`) or multipart field `file`. Unknown cameras are created on the fly. Returns the camera status. |
| GET | `/sites` | Aggregate per site: `slot_count`, `free`, `occupied`, `unknown`, `free_slot_ids`, per-camera summary. |
| GET | `/cameras` | All cameras with layout/online state. |
| GET | `/cameras/{camera_id}/status` | Latest per-slot status (see below). |
| GET | `/cameras/{camera_id}/history?limit=100` | Last results (free/occupied per frame) for charts. |
| GET | `/cameras/{camera_id}/preview.jpg` | Blurred, annotated frame (≤ 800 px wide). |
| GET | `/cameras/{camera_id}/calibration-frame.jpg` | Blurred frame at working resolution for the admin editor (polygons map 1:1). |
| GET | `/cameras/{camera_id}/layout?pixels=true` | Layout; normalised 0..1 by default, pixel coordinates with `pixels=true`. |
| PUT | `/cameras/{camera_id}/layout?pixels=true` | Admin saves rows (4 corners + `slot_count`) and/or slots and sets `status: "confirmed"`. |
| POST | `/cameras/{camera_id}/calibrate?use_vlm=true` | Stage A on the buffered frames; stores the proposal as `status: "proposed"`. |
| DELETE | `/cameras/{camera_id}/layout` | Forget the layout. |
| GET | `/health` | Model, GPU, VLM state, pull and forward (push to the app) state. |

### CameraStatus (GET `/cameras/{id}/status`)

```json
{
  "camera_id": "cam01", "site_id": "site01", "online": true, "frame_age_s": 2.4, "updated_at": "2026-10-31T12:00:00Z",
  "layout_status": "confirmed", "occupancy_mode": "per_slot", "estimated": false,
  "frame_size": [1280, 853], "slot_count": 9, "free": 3, "occupied": 6, "unknown": 0,
  "slots": [
    {"slot_id": "cam01-S01", "index": 1, "status": "occupied", "confidence": 1.0, "score": 1.0,
     "last_change": "2026-10-31T11:58:00Z", "polygon": [[166.7, 241.7], [254.2, 248.3], [170.8, 426.7], [54.2, 412.5]]}
  ],
  "detections": {"person": 0, "bicycle": 6, "motorcycle": 0},
  "processing_ms": 101.2, "preview_url": "/cameras/cam01/preview.jpg", "message": ""
}
```

`status` is `free`, `occupied` or `unknown` (not enough frames yet, slot hidden behind a person, blurry
frame, or `count_only` mode). `polygon` is in pixels of `frame_size` (vertex order back-left, back-right,
front-right, front-left) and can be drawn on `calibration-frame.jpg`.

### Layout (GET/PUT `/cameras/{id}/layout`)

```json
{
  "camera_id": "cam02", "site_id": "site01", "site_name": "Main entrance", "frame_size": [1280, 853],
  "parking_type": "painted_bays", "status": "proposed", "source": "auto", "occupancy_mode": "per_slot",
  "rows": [{"row_id": "r1", "slot_count": 9, "front": [[0.04, 0.48], [0.97, 0.68]], "back": [[0.13, 0.28], [0.96, 0.40]], "direction": "left_to_right"}],
  "slots": [{"slot_id": "cam02-S01", "index": 1, "row_id": "r1", "polygon": [[0.13, 0.28], [0.22, 0.29], [0.13, 0.50], [0.04, 0.48]]}],
  "calibration_hint": {"source": "vlm", "parking_type": "painted_bays", "slot_count": 9, "row_direction": "left_to_right", "notes": "..."}
}
```

Admin flow in the frontend: show `calibration-frame.jpg`, draw the proposal from `layout?pixels=true`,
let the admin drag the 4 row corners and change `slot_count`, then
`PUT /cameras/{id}/layout?pixels=true` with `{"rows": [...], "status": "confirmed"}` (slots are regenerated).

## Contract for the ESP32 team

* Send **JPEG stills, not a stream**: ≥ 1280×960 (UXGA 1600×1200 works well), JPEG quality 10–12, every ~5 s.
* `POST http://<laptop-ip>:8000/cameras/<camera_id>/frames` with header `Content-Type: image/jpeg` and the
  JPEG bytes as body. The answer is the current occupancy JSON.
* Alternative (current setup): the camera serves `GET http://<esp-ip>/snapshot` (JPEG) and the service pulls
  it itself, see `BIKESLOT_PULL_CAMERAS` above - no push code needed on the ESP32.
* Fallback: keep the stock CameraWebServer `/capture` endpoint and run
  `replay.py --camera cam01 --source http://<esp-ip>/capture --interval 5 --loop` on the laptop.
* Mount the camera high, looking along the row, fixed after calibration. Open Windows Firewall port 8000.

## Privacy

* Raw frames exist only in a RAM ring buffer (last 10 per camera, for calibration) and are never written to disk.
* Persons are detected only to blur them and to mark hidden slots `unknown`; motorcycles are blurred too
  (licence plates). Previews and the single calibration frame sent to the VLM are blurred first.
* Persisted data: layouts, per-slot status history (`data/history/*.jsonl`), blurred previews. No identities,
  no tracking.
