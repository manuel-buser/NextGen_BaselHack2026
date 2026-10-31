"""Endpoints the user page reads."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api", tags=["public"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/racks")
async def racks(request: Request) -> dict:
    store = request.app.state.store
    now = time.time()
    return {"generated_at": now, "racks": store.public_view(now), "places": store.places}


def rack_or_404(request: Request, rack_id: str):
    store = request.app.state.store
    found = store.racks.get(rack_id) or store.locations.get(rack_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"unknown rack {rack_id!r}")
    return found


@router.get("/racks/{rack_id}")
async def rack(rack_id: str, request: Request) -> dict:
    store, found = request.app.state.store, rack_or_404(request, rack_id)
    view = store.location_view if found.id in store.locations else store.rack_view
    return view(found, time.time())


@router.get("/racks/{rack_id}/history")
async def rack_history(rack_id: str, request: Request, minutes: int | None = Query(None, ge=0, le=24 * 60)) -> dict:
    """Today so far, the usual day of the same type (weekday or weekend) and what it says, for one rack or location
    (its racks' counts added per minute); with `minutes`, also the free spaces expected that many minutes ahead."""
    store, found, now = request.app.state.store, rack_or_404(request, rack_id), time.time()
    view = store.history_view(found, now)
    if minutes is not None:
        view["arrival"] = store.history.arrival(found, now, view["free_now"], minutes)
    return view
