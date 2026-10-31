"""Place search for the user page, through OpenStreetMap's Nominatim.

Usage policy (https://operations.osmfoundation.org/policies/nominatim/): at most one request per second for the
whole app, a User-Agent naming the app, cached results. Searches run only on pressing search; queries stay in the
in-memory cache, never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

import httpx
from fastapi import APIRouter, HTTPException, Request

from . import __version__

USER_AGENT = f"FindMyBikeSlot/{__version__} (BaselHack 2026 bike parking demo)"
TIMEOUT_S = 5.0
MIN_GAP_S = 1.0
MAX_RESULTS = 5
MAX_QUERY = 120
CACHE_SIZE = 200
VIEWBOX_MARGIN = 0.05  # degrees, a few km around the racks

router = APIRouter(prefix="/api", tags=["public"])


class GeocoderError(Exception):
    pass


def viewbox(locations: list[dict]) -> str | None:
    """Nominatim's viewbox (lng,lat,lng,lat) around the racks: results inside it rank higher, others still come."""
    if not locations:
        return None
    lats = [p["lat"] for p in locations]
    lngs = [p["lng"] for p in locations]
    m = VIEWBOX_MARGIN
    return ",".join(f"{round(v, 4):g}" for v in (min(lngs) - m, max(lats) + m, max(lngs) + m, min(lats) - m))


def short_name(item: dict) -> str:
    return item.get("name") or str(item.get("display_name", "")).split(",")[0].strip()


class Geocoder:
    def __init__(self, url: str, box: str | None = None, transport=None, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.url = url
        self.box = box
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()  # one upstream call at a time, for the whole app
        self._last_call: float | None = None
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._http = httpx.AsyncClient(timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}, transport=transport)

    def _cached(self, key: str) -> list[dict] | None:
        found = self._cache.get(key)
        if found is not None:
            self._cache.move_to_end(key)
        return found

    async def search(self, query: str) -> list[dict]:
        key = query.casefold()
        found = self._cached(key)
        if found is not None:
            return found
        async with self._lock:
            found = self._cached(key)  # someone else may have asked the same while this one waited
            if found is not None:
                return found
            if self._last_call is not None:
                wait = self._last_call + MIN_GAP_S - self._clock()
                if wait > 0:
                    await self._sleep(wait)
            self._last_call = self._clock()
            results = await self._ask(query)
        self._cache[key] = results
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return results

    async def _ask(self, query: str) -> list[dict]:
        params = {"q": query, "format": "jsonv2", "limit": MAX_RESULTS}
        if self.box:
            params["viewbox"] = self.box
        try:
            response = await self._http.get(self.url, params=params)
            response.raise_for_status()
            items = response.json()
            return [{"name": short_name(i), "lat": float(i["lat"]), "lng": float(i["lon"])} for i in items[:MAX_RESULTS]]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise GeocoderError(type(exc).__name__) from exc

    async def aclose(self) -> None:
        await self._http.aclose()


class HideQueries(logging.Filter):
    """Drops the query string of place searches from the access log and the httpx request log."""

    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(self._hide(arg) for arg in record.args)
        return True

    def _hide(self, arg):
        text = str(arg) if isinstance(arg, (str, httpx.URL)) else None
        if text and "?" in text and text.startswith(self.prefixes):
            return text.split("?", 1)[0] + "?(hidden)"
        return arg


def hide_queries_in_logs(url: str) -> None:
    for name in ("uvicorn.access", "httpx"):
        logger = logging.getLogger(name)
        logger.filters = [f for f in logger.filters if not isinstance(f, HideQueries)]
        logger.addFilter(HideQueries(("/api/geocode", url)))


@router.get("/geocode")
async def geocode(request: Request, q: str = "") -> list[dict]:
    geocoder = request.app.state.geocoder
    if geocoder is None:
        raise HTTPException(status_code=404, detail="Place search is turned off on this server.")
    query = " ".join(q.split())
    if not query or len(query) > MAX_QUERY:
        raise HTTPException(status_code=400, detail=f"Type a place of 1 to {MAX_QUERY} characters.")
    try:
        return await geocoder.search(query)
    except GeocoderError:
        raise HTTPException(status_code=502, detail="The place search is not answering right now.") from None
