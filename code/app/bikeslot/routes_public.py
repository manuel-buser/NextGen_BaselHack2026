"""Endpoints the user page reads."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api", tags=["public"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/racks")
async def racks(request: Request) -> dict:
    store = request.app.state.store
    now = time.time()
    return {"generated_at": now, "racks": [store.rack_view(rack, now) for rack in store.racks.values()], "places": store.places}


def rack_or_404(request: Request, rack_id: str):
    found = request.app.state.store.racks.get(rack_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"unknown rack {rack_id!r}")
    return found


@router.get("/racks/{rack_id}")
async def rack(rack_id: str, request: Request) -> dict:
    return request.app.state.store.rack_view(rack_or_404(request, rack_id), time.time())


@router.get("/racks/{rack_id}/history")
async def rack_history(rack_id: str, request: Request) -> dict:
    """Today so far, the usual day of the same type (weekday or weekend) and what it says, for one rack."""
    return request.app.state.store.history_view(rack_or_404(request, rack_id), time.time())
