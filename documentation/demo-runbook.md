# Demo runbook: one live camera at Memox, 2026-11-01

What the jury sees: the user page with the racks around Elisabethenstrasse 15. The wall rack in front of the
building has four bays between concrete pillars, counted on site on 2026-10-31: 10 hooks, then 19, 19 and 19
(67 in all). Bay 2 with its 19 hooks is live from the camera; Manuel walks in with a bicycle, hangs it on a free
hook, and the free count of that bay drops on the page within about 20 seconds. Everything else on the page
(bays 1, 3 and 4, the other addresses) is demo data and labelled as such.

## What runs where

| Part | Where | Port | Code |
|---|---|---|---|
| Camera (XIAO ESP32) | on the wall, phone hotspot | - | blurs faces and plates, pushes the JPEG every 10 s (`SNAPSHOT_INTERVAL_S`) and its health to the relay |
| Relay | Manuel's VPS, `https://relay.manuel-buser.com` | 443 | `relay/`: keeps the newest picture per camera in memory, serves it with the view key |
| Vision service | Manuel's laptop | 8000 | `code/bikeslot`: asks the relay for a new picture every second, classifies the 19 hooks of bay 2, serves `GET /cameras/live01/status` |
| App (user + admin page) | same laptop, or the frontend laptop, or a server | 8080 locally | `code/app`: reads the decided hooks from the vision service (rack `MX-CAM`), the rest is demo data |
| Demo driver | wherever the app is | - | `code/app/tools/demo_data.py` feeds the demo racks |

The app never receives a picture on this path. It asks the vision service for the decided hooks (pull) or the
service pushes them (for an app on a server). Details: `code/app/README.md`, "Live rack from the vision service",
and `documentation/CONTRACTS.md`, section 2b.

## The evening before

1. Pull the repo on every laptop that takes part. In `code/app`: `uv sync`. On the vision laptop the venv in the
   repo root (`.venv`) is already set up.
2. Rehearse without the camera (any laptop, three terminals in `code/app`):
   ```sh
   uv run uvicorn tools.fake_vision:app --port 8020
   BIKESLOT_RACKS=racks.memox.yaml VISION_POLL_S=2 uv run uvicorn bikeslot.main:app --port 8080
   uv run python tools/demo_data.py --app http://127.0.0.1:8080 --inactive none --warmup 7
   ```
   For this rehearsal change the `url` of rack `MX-CAM` in `racks.memox.yaml` to `http://127.0.0.1:8020` (and back
   afterwards). Open http://127.0.0.1:8080 and http://127.0.0.1:8080/admin. Then play the live moment:
   `curl -X POST http://127.0.0.1:8020/fake/park/7` (a bike at hook 7), `curl -X POST http://127.0.0.1:8020/fake/leave/7`.
3. The hook counts are in `racks.memox.yaml` (`MX-1` 10, `MX-CAM` 19, `MX-3` 19, `MX-4` 19). Which side of the
   camera bay 1 lies on is still a guess (north); swap the coordinates of `MX-1` and `MX-3`/`MX-4` if it is wrong.

## At the venue, in this order

1. **Network.** The camera runs on a phone hotspot and pushes its blurred pictures to the relay
   (`https://relay.manuel-buser.com`, `relay/README.md`), so the vision laptop only needs internet. The app's
   laptop must reach the vision laptop (same WiFi), or the app runs on a server (see below). Note the vision
   laptop's address (`ipconfig`, the WiFi adapter's IPv4). Windows Firewall on the vision laptop must allow
   inbound TCP 8000 (it asks on the first start of uvicorn; answer "allow" for private networks). Check the relay
   first: `https://relay.manuel-buser.com/?key=<VIEW_KEY>` shows the live picture, `/health?key=<VIEW_KEY>` says
   `"online": true`.
2. **Vision service** (repo root on the vision laptop). The relay address and the view key are in `.env` (copy
   `.env.example`, fill in the key; `.env` is never committed):
   ```sh
   # .env
   BIKESLOT_PULL_CAMERAS=live01=https://relay.manuel-buser.com/snapshot
   BIKESLOT_PULL_VIEW_KEY=<VIEW_KEY>
   ```
   ```sh
   .venv/Scripts/python -m uvicorn --app-dir code bikeslot.api:app --host 0.0.0.0 --port 8000
   ```
   The log must say `demo slot layout also loaded for live01 (19 slots ...)` and `pulling live01 from
   https://relay.manuel-buser.com/snapshot + /health every 1 s (X-View-Key set)`. Check http://127.0.0.1:8000/
   (debug view: the live picture with the 19 coloured hooks) and http://127.0.0.1:8000/cameras/live01/status
   (`slot_count` 19, `free` and `occupied` plausible, `online` true, `source.kind` relay). `GET /health` lists the
   puller under `pull`: `state` ok / not_modified (nothing new since the last poll) / waiting (the camera has not
   sent anything yet) / stale (the relay's newest picture is older than `BIKESLOT_PULL_MAX_AGE_S`, 60 s) / error
   (`last_error` says why; 401 = wrong view key). Camera on the LAN instead: `BIKESLOT_PULL_CAMERAS=live01=
   http://192.168.21.95/snapshot` and no key.
3. **Mount check.** The 19 polygons were drawn on `assets/pictures/DemoPictures/demo_picture.jpeg`. Put the camera
   where that picture was taken: the two pillars at the left and right edge of the hooks, the whole row in view.
   From that angle hook 3 is hidden behind a bike hanging on hook 4, so the count can be one low then; a camera
   looking at the bay head-on sees all 19. If the coloured polygons do not sit on the hooks, redraw them once
   (about two minutes). Head-on: `slotcls_hooks.py --images data/live01_frame.jpg --camera live01 --slots 19`
   finds the row by itself (overlay next to the file). At an angle, click the four corners:
   ```sh
   curl -s "https://relay.manuel-buser.com/snapshot?key=<VIEW_KEY>" -o data/live01_frame.jpg
   .venv/Scripts/python code/scripts/slotcls_calibrate.py --image data/live01_frame.jpg --camera live01 --slots 19 --interactive --out data/layouts/live01.json
   curl -X PUT -H "Content-Type: application/json" --data-binary @data/layouts/live01.json http://127.0.0.1:8000/cameras/live01/layout
   ```
   (click the four corners of the row of hooks: top-left, top-right, bottom-right, bottom-left; the overlay JPEG
   next to the layout shows the result; the PUT applies it without a restart, and the file is loaded on later
   starts too). The classifier was trained on the day picture of the camera and on the night pictures of the
   evening before (`slot-classifier.md`); if hooks are misjudged in tomorrow's light, label a few snapshots and
   retrain (steps 2 and 3 there), the service picks up the new model on restart.
4. **App** (in `code/app`, on whichever laptop shows the pages; put the vision laptop's address into the `url`
   of rack `MX-CAM` in `racks.memox.yaml` first):
   ```sh
   BIKESLOT_RACKS=racks.memox.yaml STALE_AFTER_S=30 uv run uvicorn bikeslot.main:app --host 0.0.0.0 --port 8080
   uv run python tools/demo_data.py --app http://127.0.0.1:8080 --inactive none --warmup 7
   ```
   On Windows PowerShell: `$env:BIKESLOT_RACKS="racks.memox.yaml"; $env:STALE_AFTER_S="30"; uv run uvicorn ...`.
   With Docker instead: `APP_PORT=8080 BIKESLOT_RACKS=racks.memox.yaml docker compose up --build` (the app in the
   container reaches the vision laptop by its LAN address, not 127.0.0.1).
5. **Check** http://127.0.0.1:8080/admin: the card of `MX-CAM` says "Vision service, camera live01, asked every
   3 s at http://...: 14 free, 5 taken of 19 hooks, frame 2 s ago" and shows the live preview; its chips are
   "Online" and "Live". On the user page the rack "Memox, bay 2 (camera)" shows the same free count with the
   "Real camera" chip; bays 1, 3 and 4 and the other addresses carry "Demo data".
6. **Phones.** Open `http://<app-laptop-address>:8080` on the phones that will be shown; the page refreshes
   itself every 5 s.

## The live moment

- Before: say the number on the page ("14 of 19 free in bay 2").
- Walk in with the bike, hang it on a free hook, step out of the picture. The camera pushes a new picture every
  10 s, the service picks it up within a second and needs 3 of its last 5 pictures to agree (about 20 to 30 s),
  the app shows the new count at its next poll (3 s).
- After: the count is one lower, the hook is red in the rack plan (tap the rack on the user page, or zoom in on
  the map). The admin page's preview shows the same hook red in the camera picture, persons blurred.
- Taking the bike away works the same way: free again after about 20 to 30 s.

Faster reaction, at the price of more flicker, on the vision laptop before the start:
`set BIKESLOT_K_FRAMES=3` (2 of 3 pictures agree: about 20 s). The picture rate itself is set on the camera
(`SNAPSHOT_INTERVAL_S` in its `config.h`, 10 s = about 16 MB of mobile data per hour); polling faster than
1 s gains nothing.

## If something is off

| Symptom | Where to look | Fix |
|---|---|---|
| `MX-CAM` card: "nothing received yet", error "ConnectError" | the app cannot reach the vision laptop | address in `racks.memox.yaml`, same WiFi, firewall port 8000 on the vision laptop; `curl http://<laptop>:8000/health` from the app's laptop |
| error "camera 'live01' is not known to the service yet" | the service has no frame yet | the camera address in `BIKESLOT_PULL_CAMERAS`; `GET /health` of the service shows `pull` errors |
| card says "no slot layout on the service yet" | the layout was not seeded for `live01` | start log; `BIKESLOT_DEMO_LAYOUT_CAMERAS=live01` is the default, or redraw (step 3) |
| hooks right in the preview, count wrong on the page | numbering | the app shows the service's states as they are; compare `GET /cameras/live01/status` with `GET /api/racks/MX-CAM` |
| rack shows "No recent data" | no new decided frame for `STALE_AFTER_S` | camera offline or the service stuck: the service's `/health` (`pull.last_error`) |
| the service's `/health` says `pull.state` waiting | the relay has no picture from the camera yet | the camera and its hotspot: `https://relay.manuel-buser.com/health?key=<VIEW_KEY>` must say `online` |
| `pull.state` error, `last_error` "view key refused (401)" | wrong or missing key | `BIKESLOT_PULL_VIEW_KEY` in `.env` = the relay's `VIEW_KEY` |
| rack grey ("Not tracked live") although the service runs | the relay reports the camera offline, or its newest picture is older than 60 s | the camera stopped pushing: hotspot, power; the status `message` and `source` say which |
| the service misjudges a hook | model | label a few snapshots and retrain (`slot-classifier.md`); until then say so, the count is what it is |
| the app runs on a server and cannot reach the laptop | push instead of pull | on the laptop `set BIKESLOT_APP_URL=https://<server>` and `set BIKESLOT_APP_RACKS=live01=MX-CAM` (and `BIKESLOT_APP_TOKEN` = the server's `VISION_TOKEN`); in the server's `racks.memox.yaml` drop the `url` of `MX-CAM`. The preview is not available on the admin page then |

Fallback without any camera: run `tools.fake_vision:app` as in the rehearsal and point `MX-CAM` at it; the page then
says "Real camera" for a stand-in, so only as a last resort and said out loud.
