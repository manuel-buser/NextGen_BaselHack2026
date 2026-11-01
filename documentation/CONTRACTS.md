# Interfaces between the three parts

Three parts, two interfaces. The app sits in the middle. Everything is plain HTTP with JSON, so each
part can be built and tested alone. Reference stand-ins for both sides live in `code/app/tools/`.
A third path (section 2b) is used for the camera kept by the vision laptop: there the vision service holds the
camera and the per-hook state itself, and the app reads the decided spaces instead of receiving pictures.

```
 camera node  --JPEG + alive-->  app  --JPEG-->  detector
                                  ^                |
                                  +-- JSON result --+
                                  user page: free per rack
                                  admin page: which parts are alive
```

## 1. Camera node to app

Two options. The node implements one of them; the app supports both, chosen per rack in its racks file.
- Option A, push: the node posts pictures and alive messages to the app. The firmware needs the app's address.
- Option B, pull: the node serves a status JSON and a picture, the app fetches both. The app's racks file needs
  the node's address (`camera: {url: ...}`), the firmware needs nothing about the app.

### Option A: the node posts (push)

Base URL: the laptop running the app, for example `http://192.168.43.10:8000`. Rack id: configured in the
sketch, must match `code/app/racks.yaml` (`A1` to `A4`). Optional header `X-Node-Id: esp32-1` on every call.
Through a public address (a tunnel such as `https://abc.trycloudflare.com`) the app is read-only: every POST gets
`403`. Post to the laptop's IP address or to `localhost`.

### `POST /api/camera/{rack}/picture`
- Body: the raw JPEG bytes. Header `Content-Type: image/jpeg`.
- Send one every 5 to 10 seconds for the demo (60 seconds in real use). 1024x768 or 1280x1024 is plenty.
- Reply `200 {"ok": true, "rack": "A2", "free": 7, "capacity": 19, "status": "ok"}`.
- `503 {"ok": false, "error": "detector unavailable"}`: the recognition service is down; keep sending.
- `404`: unknown rack id. `400`: empty body.
- The app keeps the picture only for this request and never writes it to disk.

### `POST /api/camera/{rack}/alive`
- Body JSON: `{"tempC": 41.2, "tooHot": false, "uptimeS": 1234}`. All fields optional. Header
  `Content-Type: application/json`.
- Send one with every picture, and keep sending it while paused for heat (the admin page then shows
  "too hot" instead of "offline").
- Reply `200 {"ok": true}`.

Try it without hardware: `python code/app/tools/fake_camera.py --app http://localhost:8000 --rack A2`.

Example with curl:
```sh
curl -X POST --data-binary @photo.jpg -H "Content-Type: image/jpeg" http://localhost:8000/api/camera/A2/picture
curl -X POST -d '{"tempC":41.2,"tooHot":false}' -H "Content-Type: application/json" http://localhost:8000/api/camera/A2/alive
```

### Option B: the app polls the node (pull)

The rack's entry in `code/app/racks.yaml` names the node:
`camera: {url: "http://192.168.21.95", status_path: "/status", picture_path: "/capture", poll_s: 5, picture_every_s: 10}`
(only `url` is required, the rest are the defaults shown). The app then makes two kinds of request:

- `GET {url}{status_path}` every `poll_s` seconds (5), timeout 3 seconds. Reply `200` with a JSON object. A
  successful status poll counts as alive: the camera shows online on the admin page while these polls succeed
  and offline once they have failed for `STALE_AFTER_S`. Fields the app reads (all optional, other fields are
  ignored); the example values are a real reply from the node firmware:

  | Field | Example | What the app does with it |
  |---|---|---|
  | `temperature_c` | `51.3` | chip temperature on the admin page (`tempC` is accepted too) |
  | `too_hot` | `false` | requested from the firmware, not confirmed yet: taken as is when present; when missing the app derives it (temperature above `HOT_LIMIT_C`, default 75) and marks it as derived |
  | `uptime_s` | `155` | shown; a value below the previous reading is recorded as a reboot, shown for 10 minutes |
  | `wifi_rssi_dbm` | `-38` | shown with a word: excellent from -60 up, ok from -75 up, weak below |
  | `wifi_connected`, `wifi_channel` | `true`, `5` | kept in the admin API |
  | `heap_free`, `heap_min_free` | `202840`, `198216` | kept in the admin API |
  | `heap_largest_block` | `159732` | shown in KB with a warning under 100 KB (a JPEG needs one contiguous block) |
  | `psram_free`, `psram_total` | `7987060`, `8388608` | kept in the admin API |
  | `stream_active`, `stream_frames`, `stream_dropped` | `false`, `0`, `0` | frames and dropped kept in the admin API, `stream_active` ignored |
  | `ip` | `"192.168.21.95"` | kept in the admin API, not shown on the page |

- `GET {url}{picture_path}` every `picture_every_s` seconds (10), timeout 10 seconds, only after a successful
  status poll. Reply `200` with the raw JPEG bytes. A still-picture endpoint has been requested from the
  firmware and is not confirmed yet; the path is configurable for that reason. The picture goes through
  exactly the same path as a posted one: detector, rules, then discarded.
- The node id shown on the admin page is the host part of `url`. A failed request is shown per camera on the
  admin page and the next poll tries again; nothing stops.

Try it without hardware: `uvicorn tools.fake_node:app --port 8010` from `code/app`, then
`camera: {url: "http://localhost:8010"}` on a rack. `CAMERA_PULL=0` turns all polling off.

## 2. App to detector

The app is configured with `DETECTOR_URL`, for example `http://localhost:8001`. The detector is stateless: one
picture in, one result out. The app applies the time rules (free only after three clean pictures, etc.).
Racks marked `demo: true` in the racks file are made up for the demo; the app decides their pictures with its
own stand-in and never sends them to the detector.

### `GET /health`
- Reply `200 {"status": "ok", "model": "yolo11n", "version": "0.1.0"}`. Shown on the admin page.

### `POST /analyze?rack=A2`
- Body: the raw JPEG bytes, header `Content-Type: image/jpeg`. Query parameter `rack` tells the detector
  which slot boxes to use (its own config, for example `slots/A2.json`).
- Reply `200` with:
```json
{
  "rack": "A2",
  "slots": [
    {"id": 1, "state": "taken", "confidence": 0.93},
    {"id": 2, "state": "free", "confidence": 0.88},
    {"id": 3, "state": "unsure", "confidence": 0.35}
  ],
  "bike_count": 7,
  "bikes": [{"x": 120, "y": 340, "w": 210, "h": 160, "confidence": 0.91}],
  "model": "yolo11n",
  "ms": 120
}
```
- `slots` is the preferred form (one entry per parking space, `state` is `taken`, `free` or `unsure`).
  If the team ends up with plain counting, leave `slots` out and send `bike_count` only; the app handles both.
- Slot numbering: the ids are 1 to the rack's capacity and follow the rack plan in the app's racks file
  (`layout`): spaces are numbered from the entrance along the rack (along the layout's bearing), row by row,
  row 1 being the left row when you look along the rack from the entrance. The app draws space n where the
  layout puts it, so a slot box drawn for the wrong number shows the wrong space as free. The section with
  the camera (A2 for now) has 19 spaces in one row, numbered 1 to 19 from the west end.
- `bikes` is optional and only for debugging; the app ignores it. Never include person detections.
- Any non-200 reply or timeout makes the app answer the camera with 503 and shows the detector as down on the
  admin page. Default timeout 15 seconds.

Reference implementation with the exact shape: `code/app/tools/fake_detector.py`
(`uvicorn tools.fake_detector:app --port 8001` from `code/app`). Replace the body of `analyze()`.

## 2b. Vision service to app (decided spaces, no pictures)

For a rack with a `vision` block in `code/app/racks*.yaml` (`vision: {url: "http://<vision laptop>:8000",
camera: live01}`) the vision service (`code/bikeslot`, `documentation/vision-pipeline.md`) keeps the camera:
it pulls the snapshots, classifies every hook, smooths over a few frames and serves the result. The app only
reads that result. Two directions, the body is the same `CameraStatus` JSON in both:

- **Pull** (the app asks): `GET {url}/cameras/{camera}/status` every `VISION_POLL_S` seconds (3), timeout 5 s.
  The admin page also fetches `GET {url}/cameras/{camera}/preview.jpg` through the app.
- **Push** (the service posts, for an app the laptop can reach but that cannot reach the laptop): `POST
  /api/vision/{rack}/status` with the JSON body, header `Content-Type: application/json` and, when the app has
  `VISION_TOKEN` set, `X-Vision-Token: <token>`. On the laptop: `BIKESLOT_APP_URL=http://<app>`,
  `BIKESLOT_APP_RACKS=live01=MX-CAM`, `BIKESLOT_APP_TOKEN=<token>`. Replies: `200 {"ok": true, "rack": "MX-CAM",
  "free": 13, "capacity": 18, "status": "ok", "applied": true}` (`applied` false when the status carried no new
  frame or nothing decided yet), `400` not a status object, `401` token, `404` unknown rack, `409` the rack has no
  `vision` block.

The fields the app reads:
```json
{"camera_id": "live01", "online": true, "updated_at": "2026-11-01T08:00:05.123+00:00", "frame_age_s": 2.1,
 "layout_status": "confirmed", "slot_count": 18, "free": 13, "occupied": 5, "unknown": 0, "processing_ms": 92.0,
 "message": "", "preview_url": "/cameras/live01/preview.jpg",
 "slots": [{"slot_id": "live01-S01", "index": 1, "status": "free"}, {"slot_id": "live01-S03", "index": 3, "status": "occupied"}]}
```
- `slots[].index` is the space number 1 to the rack's capacity (hooks numbered left to right in the picture);
  `status` is `free`, `occupied` or `unknown`. A new `updated_at` means a new decided frame; the same value again
  is only a sign of life.
- The app takes `free` and `occupied` as decided (the service has applied its own window and hysteresis; the
  app's `FREE_AFTER_CLEAN` and `MIN_CONFIDENCE` rules do not apply to this path) and leaves `unknown` hooks as
  the app last saw them. A rack without a new decided frame for `STALE_AFTER_S` shows "stale".
- Slot indexes outside 1..capacity, or a slot count that differs from the capacity, are shown as an error on the
  admin page; the fitting slots are still applied.
- `online: false` (the service has no recent picture: the camera stopped sending, the relay reports it offline or
  holds only an old picture) and a `frame_age_s` above `STALE_AFTER_S` both make every space count as occupied
  and the rack show as not tracked live, whatever `slots` says: an old picture is never shown as free spaces.
- `source` (optional) is what the service knows about where its pictures come from: `kind` (`relay` or
  `camera`), `online`, `last_seen_s`, `snapshot_age_s`, `state` (`ok`, `not_modified`, `waiting`, `stale`,
  `error`), `last_error`, and the camera's `temperature_c`, `wifi_rssi_dbm`, `wifi_connected`, `uptime_s`. The
  admin page shows them on the camera card (temperature gauge, WiFi bars, uptime, relay line).
- Other fields of the status are ignored. Nothing about a person is in it; the service blurs persons in its own
  preview before the app can fetch it.

Stand-in with the exact shape: `code/app/tools/fake_vision.py` (`uvicorn tools.fake_vision:app --port 8020` from
`code/app`; `POST /fake/park/7`, `POST /fake/leave/7` move a bike for a rehearsal).

## 3. App to pages (for reference)

- `GET /api/racks` -> `{"generated_at": ..., "racks": [{"id": "B", "name": "...", "capacity": 12, "free": 7,
  "status": "ok|starting|stale|unknown", "mode": "slots|count", "updated_at": ..., "age_s": 4.2,
  "camera_too_hot": false, "location": {"lat": 47.5472, "lng": 7.5969} or null, "demo": false,
  "source": "camera|vision|demo",
  "layout": {"bearing": 120, "entrance": "start|end", "width": 6, "height": 2,
  "spaces": [{"id": 1, "x": 0, "y": 0, "w": 1, "h": 1}, ...]} or null,
  "slots": [{"id": 1, "state": "taken|free|unknown"}, ...],
  "forecast": {"usually_full_by": "08:15" or null, "usual_free_now": 3.4 or null, "days": 4, "learning": false}}]}`
  (live slot states only; `forecast` comes from the free counts over time, see the history below;
  `demo` is true for a made-up rack fed by `code/app/tools/demo_data.py`; `layout` is the rack plan in grid
  units, x along the bearing from space 1 and y across it, the same for both `layout` forms of the racks file).
  Racks with the same `group` in the racks file come as one location entry in the place of its first rack:
  `{"id": "Peter Merian-Weg", "name": "Peter Merian-Weg", "capacity": 67, "free": 40, "status": ..., "location":
  the middle of its racks, "demo": false unless all its racks are, "layout": null, "slots": [], "forecast": ...,
  "parts": [one rack each in the shape above, without "forecast"]}`. `free` adds up the racks that are "ok" (a
  rack without live data adds none) and is null when none is; `status` is "ok" when any rack is.
- `GET /api/racks/{id}` -> one rack or location in the same shape; 404 for an unknown id.
- `GET /api/racks/{id}/history` -> `{"rack": "A2", "now": "07:40", "day_type":
  "weekday|weekend", "free_now": 6 or null, "capacity": 19, "today": [{"t": "06:00", "free": 12.0}, ...],
  "typical": [{"t": "00:00", "free": 14.2 or null, "full_share": 0.0, "n": 12}, ...], "insights":
  {"usually_full_by": "08:15" or null, "best_before": "07:45" or null, "usual_free_now": 3.4 or null, "days": 4,
  "learning": false}}`; 404 for an unknown rack. For a location the free counts of its racks are added per
  minute (a rack without a sample in that minute adds none). Only free counts per rack are kept, one sample per minute while
  the rack is "ok". `now`, `day_type` and the insights use the app's local time, the same for every rack. `today`: 15-minute averages of today so far, oldest first. `typical`: always 96 entries of
  15 minutes for this day type, `free` null where there is no data, `n` the sample count. `free_now` is null
  unless the rack is "ok". `usually_full_by`: the first slot full in at least half of its samples over at least
  2 days; `best_before`: the last slot before it with at least 2 free on average; `days`: days learned for this
  day type; `learning`: fewer than 2 days. `?minutes=12` (0 to 1440, else 422) adds `"arrival": {"minutes": 12,
  "at": "07:52", "free": 9 or null}`: the usual free count at `at` (its own day type) plus `free_now` minus the
  usual count now, that difference halved per 60 minutes ahead, clamped to 0..capacity and rounded; `free_now`
  when either usual count is missing; null when `free_now` is.
- `GET /api/admin/status` -> app, detector (reachable, model, last error), cameras (`source` push, pull or
  vision, last alive, last picture, temperature, too hot and whether it was derived, last error, `node` with
  uptime, reboot time, WiFi, memory and ip, and for a vision rack `vision` with the service's url, camera, `how`
  pull or push, its own counts, frame age, message, warning and whether a preview is available), racks.
- `GET /api/health` -> `{"status": "ok"}`.
