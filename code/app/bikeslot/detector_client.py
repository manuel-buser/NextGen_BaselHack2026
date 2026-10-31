"""Client for the recognition service, with a built-in stand-in when DETECTOR_URL=fake."""

from __future__ import annotations

import time

import httpx

from .config import Settings
from .state import FREE, TAKEN


class DetectorError(Exception):
    pass


DEMO_PREFIX = b"DEMO:"


def fake_result(rack_id: str, capacity: int, image: bytes) -> dict:
    # A made-up picture may spell out its spaces ("DEMO:1101", 1 = bike seen); otherwise the picture length
    # decides the taken count, so test senders can steer the result.
    if image.startswith(DEMO_PREFIX):
        pattern = image[len(DEMO_PREFIX) : len(DEMO_PREFIX) + capacity].decode("ascii", "replace")
        seen = [char == "1" for char in pattern.ljust(capacity, "0")]
    else:
        taken = len(image) % (capacity + 1)
        seen = [i < taken for i in range(capacity)]
    slots = [{"id": i + 1, "state": TAKEN if s else FREE, "confidence": 0.9} for i, s in enumerate(seen)]
    return {"rack": rack_id, "slots": slots, "bike_count": sum(seen), "bikes": [], "model": "fake", "ms": 0}


class DetectorClient:
    def __init__(self, settings: Settings) -> None:
        self.url = settings.detector_url.rstrip("/")
        self.fake = self.url == "fake"
        self._http = (
            None if self.fake else httpx.AsyncClient(base_url=self.url, timeout=settings.detector_timeout_s)
        )

    async def analyze(self, rack_id: str, capacity: int, image: bytes) -> tuple[dict, float]:
        started = time.perf_counter()
        if self.fake:
            result = fake_result(rack_id, capacity, image)
        else:
            try:
                response = await self._http.post(
                    "/analyze",
                    params={"rack": rack_id},
                    content=image,
                    headers={"Content-Type": "image/jpeg"},
                )
                response.raise_for_status()
                result = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise DetectorError(f"{type(exc).__name__}: {exc}") from exc
        return result, (time.perf_counter() - started) * 1000

    async def health(self) -> dict:
        if self.fake:
            return {"status": "ok", "model": "fake"}
        try:
            response = await self._http.get("/health", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DetectorError(f"{type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
