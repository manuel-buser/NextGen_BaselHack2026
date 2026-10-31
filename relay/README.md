# Camera relay

A small web service in the cloud between the camera and whoever wants to see it:

```
XIAO camera ──Wi-Fi──► phone hotspot ──4G──► relay (HTTPS) ◄── browser / presentation laptop / bikeslot server
              POST /ingest/snapshot, /ingest/health          GET /snapshot, /health, /
```

It keeps **only the newest** snapshot and health message per camera, **in memory**. Nothing is
written to disk and there is no history. Faces and licence plates are already blurred on the
camera before anything is sent.

## Endpoints

| Endpoint | Who | Auth | What |
|---|---|---|---|
| `POST /ingest/snapshot` | camera | `X-Upload-Token` | Body: raw JPEG. Headers: `X-Camera-Id`, optional `X-People-Blurred`, `X-Plates-Blurred`, `X-Detect-Ms`. Answers `204` |
| `POST /ingest/health` | camera | `X-Upload-Token` | Body: JSON object, header `X-Camera-Id`. Answers `204` |
| `GET /snapshot` | viewer | view key | Latest JPEG, headers `X-Snapshot-Age-S`, `X-Camera-Id`, blur counts. `ETag` / `If-None-Match` → `304` when unchanged |
| `GET /health` | viewer | view key | `online`, `last_seen_s`, counters, and the camera's last health JSON |
| `GET /cameras` | viewer | view key | Status of all cameras |
| `GET /` | browser | view key | Viewer page: picture + health, refreshes itself |
| `GET /healthz` | host | none | Liveness of the relay itself (no camera data) |

View key: `?key=...` or header `X-View-Key`. `?camera=<id>` picks a camera; without it, the one heard
from most recently.

Errors: `401` wrong token/key, `400` bad camera id or JSON, `413` too large, `415` not a JPEG,
`429` too many cameras, `404` nothing received yet.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `UPLOAD_TOKEN` | **required**, ≥ 16 chars | The camera sends it. The relay refuses to start without it |
| `VIEW_KEY` | empty = open | If set, reading needs it |
| `ONLINE_TIMEOUT_S` | `30` | Camera shows as offline after this long without data |
| `MAX_SNAPSHOT_BYTES` | `2000000` | Larger uploads are rejected |
| `MAX_CAMERAS` | `16` | Distinct camera ids accepted |
| `PORT` | `8000` | Set automatically by Render |

Never put the token or key in the code, the Dockerfile or the repo; set them at the host.

## Deploy on Render (free, from GitHub)

1. Push this repo to GitHub (`render.yaml` is in the repo root).
2. Render dashboard → **New → Blueprint** → pick the repo → **Apply**.
   Render builds `relay/Dockerfile` and **generates** `UPLOAD_TOKEN` and `VIEW_KEY`.
3. Service → **Environment**: copy `UPLOAD_TOKEN` into the camera's `config.h`, note `VIEW_KEY`.
4. Check: `https://<service>.onrender.com/healthz` → `{"ok":true,...}`.
5. Viewer: `https://<service>.onrender.com/?key=<VIEW_KEY>`.

Every push to `main` redeploys. The free plan sleeps after 15 min without requests; the camera's
uploads keep it awake. A redeploy or restart empties the memory until the camera's next upload.

**Run exactly one instance**: the newest picture lives in the process memory.

## Other ways to run it

```bash
# Docker Hub
docker build -t <user>/bikeslot-relay:1.0 .
docker push <user>/bikeslot-relay:1.0        # then deploy that image on any host

# Any server with Docker
docker run -d --restart unless-stopped -p 8000:8000 \
  -e UPLOAD_TOKEN=<long-random> -e VIEW_KEY=<random> <user>/bikeslot-relay:1.0
```

Generate secrets: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
UPLOAD_TOKEN=test-token-0123456789 VIEW_KEY=vk .venv/bin/uvicorn app:create_app --factory --port 18000
```

Test the camera against a relay on your laptop (same Wi-Fi, plain HTTP is fine):
`#define RELAY_URL "http://<laptop-ip>:18000"` in `config.h`.

## Using it from the bikeslot server

Pull mode just fetches a URL, so the existing server can read from the relay unchanged:

```
PULL_CAMERAS="live01=https://<service>.onrender.com/snapshot?key=<VIEW_KEY>"
```
