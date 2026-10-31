# Find my Bike Slot: app (server, user page, admin page)

The app receives pictures from the camera nodes, hands each one to the recognition service, keeps only the
resulting counts, and serves two pages: the user page (a map with one pin per rack showing the free count;
zoomed in, the pin opens into the rack with its individual spaces; the list below repeats the counts, and a tap
on a rack shows its spaces) and the admin page (which components are alive, with a live data-flow diagram that zooms into the parts of each step). Pictures are held in memory for one
request and never written anywhere. The map needs internet for its tiles (OpenStreetMap) and the Leaflet
library; without it the list still works.

Interfaces for the camera node and the recognition service: see `documentation/CONTRACTS.md`.

## Run with Docker (nothing else to install)

```sh
cd code/app
docker compose up --build
```
Open http://localhost:8000 (user page) and http://localhost:8000/admin (admin page).
By default the app uses its built-in stand-in detector (`DETECTOR_URL=fake`). To use the real recognition
service running on the same laptop: `DETECTOR_URL=http://host.docker.internal:8001 docker compose up --build`.

## Run locally (Python 3.12 with uv)

```sh
cd code/app
uv sync                      # creates .venv with everything, including test tools
uv run uvicorn bikeslot.main:app --reload --port 8000
```
Racks are configured in `racks.yaml`: id, name, capacity, an optional `location` (lat, lng) for the map pin, an
optional `layout` for the rack plan (below), an optional `camera` block for a node the app polls (see "Pull
mode"), `demo: true` for a made-up rack (see "Demo with one real camera"), and an optional `group`: racks with
the same group are one location on the user page (one pin, one entry, the sum of their free spaces; a rack
without live data adds none), while cameras, counts, history and the admin page stay per rack. The shipped
racks are the four sections of hanging bike spaces along the Peter Merian-Weg cycle path, from the west end `A1`
(10 spaces), `A2`, `A3` and `A4` (19 each), grouped as "Peter Merian-Weg".

### Rack plan (`layout`)

With a `layout` the user page draws the rack (never a camera picture): every space at its place, green when
free, red and hatched when taken, with an entrance arrow, in the expanded tile, in the map card and on the map
itself turned to the rack's real direction. `layout: {rows: 1, cols: 15, bearing: 35}` describes a grid;
`bearing` is the compass direction in degrees clockwise from north along a row from space 1 to the last space,
and `entrance: start` (default) or `end` says which end people arrive from. `space_m: [along, depth]` is one
space in metres (default `[0.7, 1.9]`; the hanging spaces use `[0.5, 1.2]`), and `location` is the plan's centre.
Odd shapes list the spaces instead:
`layout: {bearing: 150, spaces: [{id: 1, x: 0, y: 0, w: 1, h: 1}, ...]}` in grid units, x along the bearing and
y across it (rack E in `tools/screenshots.py` has an example with a gap). Numbering convention, which the detector's
slot ids must follow: spaces are numbered from the entrance along the bearing, row by row, row 1 being the left
row when you look along the bearing from the entrance. The app checks at startup that the ids are exactly 1 to
`capacity`.
Settings are environment variables: `DETECTOR_URL` (default `fake`), `BIKESLOT_RACKS` (default `racks.yaml`),
`STALE_AFTER_S` (default 60), `FREE_AFTER_CLEAN` (default 3), `MIN_CONFIDENCE` (default 0.4), `CAMERA_PULL`
(default 1; 0 turns the polling of pull cameras off), `HOT_LIMIT_C` (default 75, see "Pull mode").

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

## See something without the hardware

In a second terminal, a stand-in camera posts a picture and an alive message every 5 seconds:
```sh
uv run python tools/fake_camera.py --app http://localhost:8000 --rack A2 --interval 5
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

`racks.demo.yaml` holds the four Peter Merian-Weg sections, of which only the one with the `camera` block
(section 2, rack A2) has the real camera, plus the made-up racks B and C; all others are marked `demo: true`. To
move the camera to another section, swap its `camera` line with that section's `demo: true` line. On the user
page a demo rack looks like any other rack (the Peter Merian-Weg location gets a "Real camera" chip for the
section with the camera), and its pictures are decided by the built-in stand-in, never by the real detector, so
the real camera and the real detector run next to it unchanged.

Start in exactly this order: the app reads the history file only when it starts and writes its own copy back
every 60 s, so the seed comes first and the app second.
```sh
rm -f data/history.json                    # 1. clear the old history: the seed never overwrites a file
uv run python tools/demo_data.py --seed    # 2. seven days of history up to now (refused while an app answers)
BIKESLOT_RACKS=racks.demo.yaml STALE_AFTER_S=30 uv run uvicorn bikeslot.main:app --port 8000   # 3. the app
uv run python tools/demo_data.py           # 4. second terminal: the live feed
```
With Docker the order is the same; the seed runs on the laptop (with the uv setup above) and writes into
`./data`, which the container reads. It uses the laptop's clock, so the laptop's time zone must match `TZ`
(default Europe/Zurich).
```sh
rm -f data/history.json
uv run python tools/demo_data.py --seed
BIKESLOT_RACKS=racks.demo.yaml STALE_AFTER_S=30 docker compose up --build
uv run python tools/demo_data.py           # second terminal
```
The seed writes, for every minute of the last seven days up to now (`--seed 3` for three days), the count of
every section that gets data, the real section A2 included, plus B, in the app's own history format. The live
feed asks the app which racks are demo and feeds them at the real time of day with the same profiles: bikes
arrive in the morning, the sections nearest the venue entrance at the west end fill first and the east end stays
emptier, bikes leave in the evening, a bike or two comes or goes every minute, and Saturdays and Sundays are
quieter (on a weekend day the demo shows a weekend). The demo rack furthest from the real camera (C) gets no data
and no history, so the page also shows an inactive rack (`--inactive ID` picks another, `--inactive none` feeds
all). Real racks are left to their cameras; `--feed-real` also feeds them for a rehearsal without the camera
(start the app with `CAMERA_PULL=0` then, also with Docker; the page does not label them as demo, so never in
front of the jury; refused while the app uses a real detector). With `--outage` the second demo rack (A3) also
loses its camera from 11:00 to 18:00 (the rack shows "No recent data" once the stale window has passed, hence the
shorter `STALE_AFTER_S`).

## Public demo in one command

`python tools/deploy.py` (Linux, WSL, macOS) or `py tools\deploy.py` (Windows PowerShell), from any folder. It
lists what is missing (uv, Python 3.12, the app packages, cloudflared) and asks once, `Install now? [Y/n]`.
Downloads are pinned and checked against their published SHA256: uv and cloudflared go to `code/app/.deploy/bin`
(with a downloaded uv, its cache and Python go to `.deploy` too), the packages to `code/app/.venv`; deleting both
folders removes everything. It then seeds a week of history if `HISTORY_FILE` (default `data/history.json`) does
not exist, starts the app on 127.0.0.1 with `racks.demo.yaml` and the demo feed (`--feed-real`, camera polling
off), opens a Cloudflare quick tunnel, opens the link in a new browser window and prints it with a QR code.
Visitors of the link can only read (see `documentation/CONTRACTS.md`). Logs go to `code/app/.deploy`.
- Ctrl+C stops the app, the feed and the tunnel. A `git pull` while it runs reloads the app and keeps the link.
- The link changes at every start: make the QR code for the slides after the final start.
- `--live`: the real camera feeds its rack, the feed only the demo racks. `--port` (default 8002, the vision
  service uses 8000), `--racks` (relative to `code/app`).
- Do not run the Docker app at the same time: both write `data/history.json`.

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
15-minute steps (`GET /api/racks/{id}/history`, and `forecast` on every rack; for a location the free counts of
its racks are added per minute first): "usually full by 08:15" once the rack was full then on at least two days, and "learning" until two days of that type are known.
- Kept in `HISTORY_FILE` (default `data/history.json` in the working folder; with Docker `./data` next to
  `compose.yaml`), saved every 60 s and at shutdown; samples older than 8 weeks are dropped. `HISTORY_FILE=`
  (empty) keeps it in memory only.
- Free on arrival: the open tile says how many spaces are expected when you get there. The page turns the
  browser's position into minutes by bike (straight line x 1.3 at 15 km/h) and sends only those
  (`GET /api/racks/{id}/history?minutes=12`); the position stays in the browser. It is read only when the
  browser already allows it, after a tap on the tile's "On arrival" or by "Nearest to me". The guess is the usual
  count then plus today's difference from the usual count now, halved per hour ahead.
- To wipe it: stop the app, delete the file (`rm data/history.json`), start again.
- Every rack, demo or real, is stamped with the app's local time (Docker: `TZ`, default Europe/Zurich).
  `tools/demo_data.py --seed` writes a week of history for every section before the app starts (see "Demo with
  one real camera"), so the pitch starts with learned days.

## Removing the developer tooling

`rm -rf ~/applications/uv ~/applications/playwright-browsers code/app/.venv`. Nothing was added to shell
startup files; `dev.env.sh` is sourced by hand.
