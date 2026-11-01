"""Endpoints of the vision path: the vision service pushes a rack's status (push mode), and the admin page fetches
the service's blurred, annotated preview through the app (pull mode; nothing is stored)."""

from __future__ import annotations

import secrets
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .poller import describe
from .state import Rack
from .vision import ingest_vision

router = APIRouter(prefix="/api/vision", tags=["vision"])
PREVIEW_TIMEOUT_S = 5.0
STAGES = ("camera", "hooks", "bikes")   # the vision service's GET /cameras/{camera}/stage/{name}.jpg


def vision_rack_or_404(request: Request, rack_id: str) -> Rack:
    rack = request.app.state.store.racks.get(rack_id)
    if rack is None:
        raise HTTPException(status_code=404, detail=f"unknown rack {rack_id!r}")
    if rack.vision is None:
        raise HTTPException(status_code=409, detail=f"rack {rack_id!r} has no vision block in the racks file")
    return rack


@router.post("/{rack_id}/status")
async def pushed_status(rack_id: str, request: Request) -> JSONResponse:
    """The vision service posts the CameraStatus JSON of this rack's camera (the same body its GET .../status serves)."""
    rack = vision_rack_or_404(request, rack_id)
    token = request.app.state.settings.vision_token
    if token and not secrets.compare_digest(request.headers.get("x-vision-token", "").encode(), token.encode()):
        raise HTTPException(status_code=401, detail="missing or wrong X-Vision-Token")
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="body must be the CameraStatus JSON of the vision service")
    body, code = ingest_vision(request.app, rack, payload, time.time())
    return JSONResponse(body, status_code=code)


async def from_vision(request: Request, rack_id: str, path: str) -> httpx.Response:
    """GET {url}/cameras/{camera}/{path} of this rack's vision service: 404 in push mode, 502 when the service fails."""
    rack = vision_rack_or_404(request, rack_id)
    if rack.vision.url is None:
        raise HTTPException(status_code=404, detail="push mode: the app has no address to fetch from")
    try:
        response = await request.app.state.vision_http.get(f"{rack.vision.url}/cameras/{rack.vision.camera}/{path}",
                                                           timeout=PREVIEW_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vision service: {describe(exc)}")
    return response


@router.get("/{rack_id}/preview.jpg")
async def preview(rack_id: str, request: Request) -> Response:
    """The service's last preview for this rack's camera: persons blurred by the service, slots drawn in colour."""
    response = await from_vision(request, rack_id, "preview.jpg")
    return Response(response.content, media_type=response.headers.get("content-type", "image/jpeg"), headers={"Cache-Control": "no-store"})


@router.get("/{rack_id}/stage/{name}.jpg")
async def stage(rack_id: str, name: str, request: Request) -> Response:
    """One step of the live processing for the admin page's Camera tab: camera, hooks (model 1) or bikes (model 2)."""
    if name not in STAGES:
        raise HTTPException(status_code=404, detail=f"unknown step {name!r}, one of {', '.join(STAGES)}")
    response = await from_vision(request, rack_id, f"stage/{name}.jpg")
    return Response(response.content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{rack_id}/stages")
async def stage_counts(rack_id: str, request: Request) -> JSONResponse:
    """Counts and timings that go with the stage pictures."""
    response = await from_vision(request, rack_id, "stages")
    try:
        return JSONResponse(response.json(), headers={"Cache-Control": "no-store"})
    except ValueError:
        raise HTTPException(status_code=502, detail="vision service: stages are not JSON")
