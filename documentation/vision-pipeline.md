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

Two stages per camera:

| Stage | When | What |
|---|---|---|
| **A – calibration** (`calibrate.py`, `stands.py`, `vlm.py`, `lines.py`) | once per camera, admin-triggered | Proposes the slot layout: how many slots, in which order, where in the image. Evidence: **rack stands** found by YOLOE visual prompting (one example stand found by text prompt, then all look-alikes; hidden stands inferred from the spacing), **parked bikes** (wheel contact points: row split, front edge, end slots), a **GPT-vision set-of-mark answer** (numbered boxes on the bikes; parking type, enumerated slots, "bike k is in slot j", only used when consistent with the bike spacing) and painted dividers as an extra count. Several rows are supported. The admin then confirms or corrects the result. |
| **B – occupancy** (`detector.py`, `occupancy.py`) | every snapshot, fully local | YOLO11x-seg finds bicycles / motorcycles / persons. Each vehicle's **ground-contact point** (lowest pixels of its mask ≈ rear wheel) and its footprint are matched against the slot polygons. One vehicle fills at most one slot. Votes are smoothed over the last 5 frames with hysteresis. |

### Why this design
* **Local detector + geometry per frame**: ~100 ms on the laptop GPU, offline, deterministic, explainable.
  The mask contact point (not the box centre) is what makes bikes leaning over a line land in the right bay.
* **VLM only at calibration**: good at "what kind of parking, how many slots, which direction", bad at
  pixel-precise coordinates, too slow/expensive per frame and needs internet.
* **Row model**: a layout is a front line + back line + slot count. The admin drags 4 corners and sets N
  instead of drawing N polygons. Slots are split equally *on the ground* (homography), so perspective is
  respected. Explicit polygons are stored too and can be edited individually.
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
| Auto layout straight from calibration (bikes + VLM) | slot count 8 of 9, front line correct, back line needs the admin to drag one corner; after setting N and the corner: 27/27 |

### Automatic calibration benchmark (2026-10-31, `scratchpad/calib_bench.py`)

Occupancy = per-slot states of the proposal applied to the same images, *without any admin correction*.
Golden truth is 9 slots; the ChatGPT images have no labels, so "bikes" is the detector count.

| Scene | Stands found | Proposed N | Occupancy with the raw proposal |
|---|---|---|---|
| golden, 3 frames with different occupancy, no VLM | 7 + 2 inferred | 11 (9 + 2 phantom end slots) | 17 / 27 |
| golden, 3 frames, with VLM | 7 + 2 inferred | 8 | 10 / 27 |
| golden, single frame | 5 + 1 inferred | 6–10 | 11–16 / 27 |
| gpt_01 (2 rows, painted bays) | 8 | 18–19 (15 bikes) | 14 of 15 bikes placed |
| gpt_02 (2 dense diagonal rows) | none | 22 (25 bikes) | 11 of 25 |
| gpt_03 (2 full rows) | none | 19 (19 bikes) | 19 of 19 |
| gpt_04 (stirrups used from both sides, diagonal) | 5 + 2 | 11 (8 bikes) | 5 of 8 |
| gpt_05 (2 full rows) | none | 18 (20 bikes) | 16 of 20 |
| gpt_06 (2 rows, 34 bikes) | none | 31 | 24 of 34 |
| gpt_07 (2 diagonal rows, painted bays) | 5 | 14 (14 bikes) | 3 of 14 |

What this means in practice:
* **Full or nearly full rows calibrate themselves** (gpt_03, gpt_05): every bike is a slot, the rows are found,
  the admin only presses confirm.
* **Stands are the strongest signal for partially used racks** (golden: 9 of 9 stirrups with three frames of
  varied occupancy). Calibrate after the camera has seen a few different occupancy states.
* **Where it still needs the admin**: end slots whose stand is hidden and whose bike leans away from the
  stand (golden gives 8 or 11 instead of 9), the depth of a far second row (gpt_07), and racks used from both
  sides (gpt_04). Those are a slot count and one corner drag in the admin editor.
* GPT-4.1's "bike k is in slot j" answers are wrong more often than right on these scenes (it tends to
  number the bikes 1..k); the consistency check rejects them in that case so they cannot do harm.
  Its enumerated slot count is a useful tie-breaker and its parking type / numbering direction are kept.
* Painted-line detection (`lines.py`) is experimental and only used as an additional count estimate.

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

# fake camera: loop the golden images into cam01 every 3 s
.venv/Scripts/python code/scripts/replay.py --camera cam01 --source assets/pictures/GoldenDataSet --interval 3 --loop

# evaluation and tests
.venv/Scripts/python code/scripts/eval_golden.py --previews data/preview/eval
cd code && ../.venv/Scripts/python -m pytest tests -q
```

`code` is also the name of a Python standard-library module, so the package is always run with
`--app-dir code` / `PYTHONPATH=code` and never imported as `code.bikeslot`.
Use `127.0.0.1`, not `localhost`, in clients on Windows (IPv6 fallback costs ~2 s per connection).

Tunables live in `code/bikeslot/config.py` and can be overridden with `BIKESLOT_*` environment variables
(e.g. `BIKESLOT_MODEL=yolo11x.pt` for the detection-only model, `BIKESLOT_CONF=0.15`, `BIKESLOT_DEVICE=cpu`).

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
| GET | `/health` | Model, GPU, VLM state. |

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
* Fallback: keep the stock CameraWebServer `/capture` endpoint and run
  `replay.py --camera cam01 --source http://<esp-ip>/capture --interval 5 --loop` on the laptop.
* Mount the camera high, looking along the row, fixed after calibration. Open Windows Firewall port 8000.

## Privacy

* Raw frames exist only in a RAM ring buffer (last 10 per camera, for calibration) and are never written to disk.
* Persons are detected only to blur them and to mark hidden slots `unknown`; motorcycles are blurred too
  (licence plates). Previews and the single calibration frame sent to the VLM are blurred first.
* Persisted data: layouts, per-slot status history (`data/history/*.jsonl`), blurred previews. No identities,
  no tracking.
