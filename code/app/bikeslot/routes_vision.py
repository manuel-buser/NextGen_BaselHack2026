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


@router.get("/{rack_id}/preview.jpg")
async def preview(rack_id: str, request: Request) -> Response:
    """The service's last preview for this rack's camera: persons blurred by the service, slots drawn in colour."""
    rack = vision_rack_or_404(request, rack_id)
    if rack.vision.url is None:
        raise HTTPException(status_code=404, detail="push mode: the app has no address to fetch the preview from")
    try:
        response = await request.app.state.vision_http.get(rack.vision.preview_url, timeout=PREVIEW_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vision service: {describe(exc)}")
    return Response(response.content, media_type=response.headers.get("content-type", "image/jpeg"), headers={"Cache-Control": "no-store"})
