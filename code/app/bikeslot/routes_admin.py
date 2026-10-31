"""Endpoint the admin page reads: which components are alive."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from . import __version__
from .detector_client import DetectorError

router = APIRouter(prefix="/api/admin", tags=["admin"])

HEALTH_CACHE_S = 5.0


async def detector_status(request: Request, now: float) -> dict:
    cache = request.app.state.detector_cache
    if now - cache.get("checked_at", 0.0) > HEALTH_CACHE_S:
        detector = request.app.state.detector
        try:
            info = await detector.health()
            cache.update(checked_at=now, reachable=True, info=info, error=None)
        except DetectorError as exc:
            cache.update(checked_at=now, reachable=False, info={}, error=str(exc))
    detector = request.app.state.detector
    return {
        "mode": "fake" if detector.fake else "http",
        "url": detector.url,
        "reachable": cache["reachable"],
        "checked_at": cache["checked_at"],
        "info": cache["info"],
        "error": cache["error"],
    }


@router.get("/status")
async def status(request: Request) -> dict:
    now = time.time()
    view = request.app.state.store.admin_view(now)
    view["app"]["version"] = __version__
    view["detector"] = await detector_status(request, now)
    return view
