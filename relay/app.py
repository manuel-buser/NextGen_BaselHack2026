"""Bike-slot camera relay.

The camera (on a phone hotspot) pushes its latest blurred snapshot and health JSON here;
anyone with the view key fetches them. Only the newest snapshot and health per camera are kept,
in memory: nothing is written to disk and there is no history.

Run a single instance (one process, one worker): the state lives in this process.

Environment:
    UPLOAD_TOKEN        required  shared secret the camera sends as X-Upload-Token
    VIEW_KEY            optional  if set, reading needs ?key=... or the X-View-Key header
    ONLINE_TIMEOUT_S    30        camera counts as offline after this long without health/snapshot
    MAX_SNAPSHOT_BYTES  2000000   larger uploads are rejected (413)
    MAX_CAMERAS         16        distinct camera ids accepted (protects memory)
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

log = logging.getLogger("relay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ----------------------------------------------------------------------------- settings

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    upload_token: str
    view_key: str
    online_timeout_s: int
    max_snapshot_bytes: int
    max_cameras: int
    max_health_bytes: int = 16_384

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("UPLOAD_TOKEN", "").strip()
        if len(token) < 16:
            # Refuse to start rather than run an upload endpoint anyone on the internet can use
            raise RuntimeError("UPLOAD_TOKEN must be set (at least 16 characters)")
        return cls(
            upload_token=token,
            view_key=os.environ.get("VIEW_KEY", "").strip(),
            online_timeout_s=_env_int("ONLINE_TIMEOUT_S", 30),
            max_snapshot_bytes=_env_int("MAX_SNAPSHOT_BYTES", 2_000_000),
            max_cameras=_env_int("MAX_CAMERAS", 16),
        )


# ----------------------------------------------------------------------------- state

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Camera headers passed through to whoever fetches the snapshot
PASS_HEADERS = ("X-People-Blurred", "X-Plates-Blurred", "X-Detect-Ms", "X-Autotune")


@dataclass
class Camera:
    camera_id: str
    jpeg: Optional[bytes] = None
    jpeg_meta: dict = field(default_factory=dict)
    snapshot_seq: int = 0          # increases with every snapshot; used as ETag
    snapshot_at: Optional[float] = None
    health: Optional[dict] = None
    health_at: Optional[float] = None
    snapshots_received: int = 0
    health_received: int = 0

    def last_seen(self) -> Optional[float]:
        times = [t for t in (self.snapshot_at, self.health_at) if t is not None]
        return max(times) if times else None


def _age(t: Optional[float], now: float) -> Optional[float]:
    return None if t is None else round(now - t, 1)


# ----------------------------------------------------------------------------- app

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    cameras: dict[str, Camera] = {}
    started_at = time.time()
    if not settings.view_key:
        log.warning("VIEW_KEY not set: snapshots and health can be read by anyone with the URL")

    app = FastAPI(title="Bike-slot camera relay", docs_url=None, redoc_url=None, openapi_url=None)

    # --- helpers ---------------------------------------------------------------

    def check_upload(request: Request) -> None:
        token = request.headers.get("x-upload-token", "")
        if not hmac.compare_digest(token.encode(), settings.upload_token.encode()):
            raise HTTPException(401, "bad or missing X-Upload-Token")

    def check_view(request: Request, key: Optional[str]) -> None:
        if not settings.view_key:
            return
        given = key or request.headers.get("x-view-key", "")
        if not hmac.compare_digest(given.encode(), settings.view_key.encode()):
            raise HTTPException(401, "bad or missing view key (?key=...)")

    def camera_for_upload(request: Request) -> Camera:
        cam_id = request.headers.get("x-camera-id", "").strip()
        if not CAMERA_ID_RE.match(cam_id):
            raise HTTPException(400, "X-Camera-Id must be 1-32 characters of A-Z a-z 0-9 _ -")
        cam = cameras.get(cam_id)
        if cam is None:
            if len(cameras) >= settings.max_cameras:
                raise HTTPException(429, "too many cameras")
            cam = cameras[cam_id] = Camera(cam_id)
            log.info("new camera %s", cam_id)
        return cam

    def camera_for_read(camera: Optional[str]) -> Camera:
        if camera:
            cam = cameras.get(camera)
            if cam is None:
                raise HTTPException(404, f"no data from camera {camera!r} yet")
            return cam
        if not cameras:
            raise HTTPException(404, "no camera has sent anything yet")
        # no camera given: the one heard from most recently
        return max(cameras.values(), key=lambda c: c.last_seen() or 0)

    async def read_body(request: Request, limit: int) -> bytes:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, f"body larger than {limit} bytes")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"body larger than {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def status_of(cam: Camera, now: float) -> dict:
        seen = cam.last_seen()
        return {
            "camera_id": cam.camera_id,
            "online": seen is not None and now - seen <= settings.online_timeout_s,
            "last_seen_s": _age(seen, now),
            "last_snapshot_age_s": _age(cam.snapshot_at, now),
            "last_health_age_s": _age(cam.health_at, now),
            "snapshots_received": cam.snapshots_received,
            "health_received": cam.health_received,
        }

    # --- camera side -----------------------------------------------------------

    @app.post("/ingest/snapshot", status_code=204)
    async def ingest_snapshot(request: Request) -> Response:
        check_upload(request)
        cam = camera_for_upload(request)
        body = await read_body(request, settings.max_snapshot_bytes)
        if len(body) < 4 or body[:2] != b"\xff\xd8":
            raise HTTPException(415, "body must be a JPEG image")
        cam.jpeg = body
        cam.jpeg_meta = {h: request.headers[h] for h in PASS_HEADERS if h in request.headers}
        cam.snapshot_seq += 1
        cam.snapshot_at = time.time()
        cam.snapshots_received += 1
        return Response(status_code=204)

    @app.post("/ingest/health", status_code=204)
    async def ingest_health(request: Request) -> Response:
        check_upload(request)
        cam = camera_for_upload(request)
        body = await read_body(request, settings.max_health_bytes)
        try:
            data = json.loads(body)
        except ValueError:
            raise HTTPException(400, "body must be JSON") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "body must be a JSON object")
        cam.health = data
        cam.health_at = time.time()
        cam.health_received += 1
        return Response(status_code=204)

    # --- viewer side -----------------------------------------------------------

    @app.get("/snapshot")
    async def snapshot(request: Request, camera: Optional[str] = None, key: Optional[str] = Query(None)) -> Response:
        check_view(request, key)
        cam = camera_for_read(camera)
        if cam.jpeg is None:
            raise HTTPException(404, f"camera {cam.camera_id!r} has not sent a snapshot yet")
        etag = f'"{cam.camera_id}-{cam.snapshot_seq}"'
        headers = {
            "Cache-Control": "no-store",
            "ETag": etag,
            "X-Camera-Id": cam.camera_id,
            "X-Snapshot-Age-S": str(_age(cam.snapshot_at, time.time())),
            **cam.jpeg_meta,
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)  # viewer already has this picture
        return Response(cam.jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/health")
    async def health(request: Request, camera: Optional[str] = None, key: Optional[str] = Query(None)) -> JSONResponse:
        check_view(request, key)
        cam = camera_for_read(camera)
        return JSONResponse({**status_of(cam, time.time()), "health": cam.health},
                            headers={"Cache-Control": "no-store"})

    @app.get("/cameras")
    async def list_cameras(request: Request, key: Optional[str] = Query(None)) -> JSONResponse:
        check_view(request, key)
        now = time.time()
        return JSONResponse([status_of(c, now) for c in cameras.values()], headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict:
        """Liveness of the relay itself (for Docker / the host), no key needed, no camera data."""
        return {"ok": True, "uptime_s": round(time.time() - started_at), "cameras": len(cameras)}

    viewer_html = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def viewer(request: Request, key: Optional[str] = Query(None)) -> HTMLResponse:
        check_view(request, key)
        return HTMLResponse(viewer_html, headers={"Cache-Control": "no-store"})

    return app
