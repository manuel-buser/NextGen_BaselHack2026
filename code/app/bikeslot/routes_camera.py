"""Endpoints called by the camera node (push mode; pull mode is bikeslot/poller.py)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .ingest import ingest_alive, ingest_picture
from .state import Rack

router = APIRouter(prefix="/api/camera", tags=["camera"])


def rack_or_404(request: Request, rack_id: str) -> Rack:
    rack = request.app.state.store.racks.get(rack_id)
    if rack is None:
        raise HTTPException(status_code=404, detail=f"unknown rack {rack_id!r}")
    return rack


@router.post("/{rack_id}/alive")
async def alive(rack_id: str, request: Request) -> dict:
    rack = rack_or_404(request, rack_id)
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    ingest_alive(request.app, rack, payload, request.headers.get("x-node-id"), time.time(), "push")
    return {"ok": True}


@router.post("/{rack_id}/picture")
async def picture(rack_id: str, request: Request) -> JSONResponse:
    rack = rack_or_404(request, rack_id)
    image = await request.body()
    if not image:
        raise HTTPException(status_code=400, detail="empty body, expected a JPEG")
    body, code = await ingest_picture(request.app, rack, image, request.headers.get("x-node-id"), time.time(), "push")
    return JSONResponse(body, status_code=code)
