"""Pull mode: one background task per rack with a `camera` block polls the node's status URL and picture URL."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from fastapi import FastAPI

from .ingest import ingest_alive, ingest_picture
from .state import Rack

log = logging.getLogger(__name__)
STATUS_TIMEOUT_S = 3.0
PICTURE_TIMEOUT_S = 10.0  # a full JPEG over the node's WiFi takes longer than a status reply


def describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


class CameraPoller:
    """Every poll_s: GET the status, a good reply counts as alive. Every picture_every_s: GET a picture and feed
    it through the same path as a posted one. Failures are recorded on the camera and the next round tries again."""

    def __init__(self, app: FastAPI, rack: Rack) -> None:
        self.app = app
        self.rack = rack
        self.pull = rack.pull
        self.picture_due_at = 0.0

    async def poll_once(self, client: httpx.AsyncClient, now: float) -> None:
        store = self.app.state.store
        try:
            response = await client.get(self.pull.status_url, timeout=STATUS_TIMEOUT_S)
            response.raise_for_status()
            payload = response.json()
            ingest_alive(self.app, self.rack, payload if isinstance(payload, dict) else {}, self.pull.node_id, now, "pull")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            store.note_camera_error(self.rack, f"status: {describe(exc)}", now)
            return  # a node that does not answer its status is not asked for a picture
        if now < self.picture_due_at:
            return
        self.picture_due_at = now + self.pull.picture_every_s
        try:
            response = await client.get(self.pull.picture_url, timeout=PICTURE_TIMEOUT_S)
            response.raise_for_status()
            if not response.content:
                raise ValueError("empty body, expected a JPEG")
        except (httpx.HTTPError, ValueError) as exc:
            store.note_camera_error(self.rack, f"picture: {describe(exc)}", now)
            return
        await ingest_picture(self.app, self.rack, response.content, self.pull.node_id, now, "pull")

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    await self.poll_once(client, time.time())
                except Exception as exc:  # a bug must not end the polling; the admin page shows it
                    self.app.state.store.note_camera_error(self.rack, describe(exc), time.time())
                    log.exception("polling rack %s failed", self.rack.id)
                await asyncio.sleep(self.pull.poll_s)
