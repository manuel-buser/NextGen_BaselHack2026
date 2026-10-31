# Find my Bike Slot: app (server, user page, admin page)

The app receives pictures from the camera nodes, hands each one to the recognition service, keeps only the
resulting counts, and serves two pages: the user page (a map with one pin per rack showing the free count;
zoomed in, the pin opens into the rack with its individual spaces; the list below repeats the counts, and a tap
on a rack shows its spaces) and the admin page (which components are alive, with a live data-flow diagram that zooms into the parts of each step). Pictures are held in memory for one
request and never written anywhere. The map needs internet for its tiles (OpenStreetMap) and the Leaflet
library; without it the list still works.

Interfaces for the camera node and the recognition service: see `documentation/CONTRACTS.md`. A rack can also take
its decided spaces from the vision service on the vision laptop instead of pictures: see "Live rack from the
vision service" below and `documentation/demo-runbook.md` for the pitch set-up.

## Run with Docker (nothing else to install)

```sh
cd code/app
docker compose up --build
```
Open http://localhost:8000 (user page) and http://localhost:8000/admin (admin page).
By default the app uses its built-in stand-in detector (`DETECTOR_URL=fake`). To use the real recognition
service running on the same laptop: `DETECTOR_URL=http://host.docker.internal:8001 docker compose up --build`.
`APP_PORT=8080 docker compose up` publishes the app on another port, for a laptop where the vision service has 8000.

## Run locally (Python 3.12 with uv)

```sh
cd code/app
uv sync                      # creates .venv with everything, including test tools
uv run uvicorn bikeslot.main:app --reload --port 8000
```
Racks are configured in `racks.yaml`: id, name, capacity, an optional `location` (lat, lng) for the map pin, an
optional `layout` for the rack plan (below), an optional `camera` block for a node the app polls (see "Pull
mode"), and `demo: true` for a made-up rack (see "Demo with one real camera").

### Rack plan (`layout`)

With a `layout` the user page draws the rack (never a camera picture): every space at its place, green when
free, red and hatched when taken, with an entrance arrow, in the expanded tile, in the map card and on the map
itself turned to the rack's real direction. `layout: {rows: 1, cols: 15, bearing: 35}` describes a grid;
`bearing` is the compass direction in degrees clockwise from north along a row from space 1 to the last space,
and `entrance: start` (default) or `end` says which end people arrive from. Odd shapes list the spaces instead:
`layout: {bearing: 150, spaces: [{id: 1, x: 0, y: 0, w: 1, h: 1}, ...]}` in grid units, x along the bearing and
y across it (`racks.demo.yaml` rack E has an example with a gap). Numbering convention, which the detector's
slot ids must follow: spaces are numbered from the entrance along the bearing, row by row, row 1 being the left
row when you look along the bearing from the entrance. The app checks at startup that the ids are exactly 1 to
`capacity`.
Settings are environment variables: `DETECTOR_URL` (default `fake`), `BIKESLOT_RACKS` (default `racks.yaml`),
`STALE_AFTER_S` (default 60), `FREE_AFTER_CLEAN` (default 3), `MIN_CONFIDENCE` (default 0.4), `CAMERA_PULL`
(default 1; 0 turns the polling of pull cameras and vision racks off), `HOT_LIMIT_C` (default 75, see "Pull mode"),
`VISION_POLL_S` (default 3) and `VISION_TOKEN` (see "Live rack from the vision service").

Place search: `GEOCODER_URL` (default `https://nominatim.openstreetmap.org/search`, empty turns it off) answers
a place typed into "Where are you going?" on the map. The browser sends the typed text to this server, which asks
OpenStreetMap's Nominatim, so the server needs internet for it. Answers stay in memory only (the last 200
searches); the text is not logged or stored. A place from the list, a tap on the map and "Nearest to me" never
leave the browser. The server keeps to the [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/):
at most one request per second for the whole app (counted per server process, so run one), a User-Agent naming
the app, cached answers, and a search only when someone presses search. For heavy use, point `GEOCODER_URL` at
your own Nominatim.

### Pull mode (`camera`)

By default a camera node posts its pictures and alive messages to the app (push). A rack with a `camera` block
is polled by the app instead: `camera: {url: "http://192.168.21.95"}` is enough; the full form is
`camera: {url: ..., status_path: "/status", picture_path: "/capture", poll_s: 5, picture_every_s: 10}` with the
defaults shown. The app fetches the status JSON every `poll_s` seconds (a good reply counts as alive) and a
picture every `picture_every_s` seconds, which then goes through the same path as a posted one. The status
fields (temperature, uptime, WiFi signal, free memory) show up on the admin page; when the node sends no
`too_hot` flag the app derives it from the temperature and `HOT_LIMIT_C`. Fetch errors are shown per camera
and the next poll tries again. Both options are described in `documentation/CONTRACTS.md`, section 1.

### Live rack from the vision service (`vision`)

The recognition service in `code/bikeslot` (the vision laptop) can keep a rack's camera itself: it fetches the
pictures from the camera, decides free or taken per hook (slot classifier, smoothed over a few frames) and serves
the result as JSON. A rack with a `vision` block takes that result instead of pictures. The app applies no rules
of its own to it (the service has smoothed already, so a change shows at the next poll), and no picture reaches
the app on this path. The pages label such a rack like any real camera.
```yaml
vision: {url: "http://192.168.21.88:8000", camera: live01, poll_s: 3}
```
- `camera`: the camera id on the vision service (its `BIKESLOT_PULL_CAMERAS`; `live01` by default).
- `url` (pull): the app asks `GET {url}/cameras/{camera}/status` every `poll_s` seconds (default `VISION_POLL_S`,
  3). A status with a new `updated_at` is a new decided frame: `free` and `occupied` become free and taken, an
  `unknown` hook (not decided yet) keeps the state the app last saw. The admin page shows the service's own counts,
  its message and its preview (persons blurred by the service, fetched through the app at
  `GET /api/vision/{rack}/preview.jpg`, never stored).
- Without `url` (push): the service posts the same JSON to `POST /api/vision/{rack}/status`; on the laptop that is
  `BIKESLOT_APP_URL=http://<this app>` and `BIKESLOT_APP_RACKS=live01=MX-CAM`. For an app on a server set
  `VISION_TOKEN`; the service sends it as `X-Vision-Token` (`BIKESLOT_APP_TOKEN`). A rack with a `url` accepts
  pushes too. A rack goes "stale" after `STALE_AFTER_S` without a new decided frame, like any other.
- `vision` cannot be combined with `camera` or `demo` on the same rack.

Without the laptop, `uv run uvicorn tools.fake_vision:app --port 8020` serves the service's JSON for camera
`live01` (19 hooks, bikes drift in and out); `curl -X POST http://127.0.0.1:8020/fake/park/7` parks a bike at hook
7 and `.../fake/leave/7` takes it away, for a rehearsal of the live moment. `racks.memox.yaml` is the set-up for
the pitch: bay 2 of the wall rack at Memox (19 hooks, the camera) comes from the vision service, the other three
bays (10, 19 and 19 hooks) and the racks at the addresses around are demo racks (`documentation/demo-runbook.md`).

## See something without the hardware

In a second terminal, a stand-in camera posts a picture and an alive message every 5 seconds:
```sh
uv run python tools/fake_camera.py --app http://localhost:8000 --rack A --interval 5
```
With `--images some/folder` it sends real JPEG files instead (use this together with the real detector).

For pull mode, run the stand-in node on a port and point a rack's `camera.url` at it:
```sh
uv run uvicorn tools.fake_node:app --port 8010
# in racks.yaml under a rack:   camera: {url: "http://localhost:8010"}
uv run uvicorn bikeslot.main:app --port 8000
```
It serves the node's status JSON (temperature drifting, uptime counting) and synthetic pictures for the
built-in stand-in detector.

To exercise the real HTTP path to a detector, run the reference detector on port 8001 and point the app at it:
```sh
uv run uvicorn tools.fake_detector:app --port 8001
DETECTOR_URL=http://localhost:8001 uv run uvicorn bikeslot.main:app --port 8000
```

## Demo with one real camera

`racks.demo.yaml` holds the real rack A plus two made-up racks marked `demo: true`. A demo rack is labelled
"Demo data" on both pages (dashed chip, dashed map pin), and its pictures are decided by the built-in stand-in,
never by the real detector, so the real camera and the real detector run next to it unchanged.
```sh
BIKESLOT_RACKS=racks.demo.yaml STALE_AFTER_S=30 uv run uvicorn bikeslot.main:app --port 8000
uv run python tools/demo_data.py          # second terminal; standard library only, plain python3 works too
```
The driver asks the app which racks are demo. It plays a compressed day in a loop (`--day 360` real seconds per
day, `--start 06:30`): bikes arrive in the morning, the first demo rack fills up, bikes leave in the evening. The
demo rack furthest from the main rack gets no data, so the page also shows an inactive rack (`--inactive ID`
picks another, `--inactive none` feeds all). Real racks are left to their cameras; `--feed-real` also feeds them
for a rehearsal without the camera (the page does not label them as demo, so never in front of the jury; refused
while the app uses a real detector). With `--outage --inactive none` the second demo rack also loses its camera from 11:00 to 18:00
(the rack shows "No recent data" once the stale window has passed, hence the shorter `STALE_AFTER_S`). Every log
line starts with the simulated clock. With Docker: `BIKESLOT_RACKS=racks.demo.yaml docker compose up --build`,
then the driver as above.

## Tests

```sh
uv run pytest                 # API and state tests
uv run playwright install chromium   # once; browsers go to PLAYWRIGHT_BROWSERS_PATH if set
uv run pytest -m ui           # browser tests (skipped automatically if Chromium cannot start)
uv run python tools/screenshots.py shots   # both pages, phone and desktop, light and dark, every rack state
```
On Ubuntu/WSL Chromium needs system libraries once: `sudo uv run playwright install-deps chromium`.

## How the numbers are formed

A slot shows "taken" as soon as the detector sees a bike in it once, and "free" only after `FREE_AFTER_CLEAN`
pictures in a row without a bike. A "free" below `MIN_CONFIDENCE`, or "unsure", counts as taken. In count mode
(detector returns only `bike_count`) the free count drops at once and rises only when the whole window agrees.
A rack is "starting" until every slot has a full window, "stale" when no result arrived for `STALE_AFTER_S`,
and "unknown" before the first result. Only counts are kept; no per-slot history is stored.

## History and prediction

While a rack shows "ok", the app keeps one sample per rack per minute: the time, the free count and the
capacity, never which space was used when. From these it learns a usual weekday and a usual weekend day in
15-minute steps (`GET /api/racks/{id}/history`, and `forecast` on every rack): "usually full by 08:15" once the
rack was full then on at least two days, and "learning" until two days of that type are known.
- Kept in `HISTORY_FILE` (default `data/history.json` in the working folder; with Docker `./data` next to
  `compose.yaml`), saved every 60 s and at shutdown; samples older than 8 weeks are dropped. `HISTORY_FILE=`
  (empty) keeps it in memory only.
- To wipe it: stop the app, delete the file (`rm data/history.json`), start again.
- Real racks use the real local time (Docker: `TZ`, default Europe/Zurich). Demo racks use the driver's
  simulated clock, which starts on Monday 2026-11-02 on every launch, so a restart overwrites the same minutes.
  `tools/demo_data.py --warmup 7` first plays the week before at full speed (a few seconds), so the pitch
  starts with learned days.

## Removing the developer tooling

`rm -rf ~/applications/uv ~/applications/playwright-browsers code/app/.venv`. Nothing was added to shell
startup files; `dev.env.sh` is sourced by hand.
