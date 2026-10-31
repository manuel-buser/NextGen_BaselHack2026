"""Find my Bike Slot server: camera endpoints, detector client, user page, admin page."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, geocoder, routes_admin, routes_camera, routes_public, routes_vision
from .config import Settings
from .detector_client import DetectorClient
from .poller import CameraPoller
from .state import Store
from .vision import VisionPoller

STATIC = Path(__file__).parent / "static"
HISTORY_SAVE_S = 60.0


async def save_history_every(history, seconds: float) -> None:
    while True:
        await asyncio.sleep(seconds)
        history.save()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    detector = DetectorClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        history = app.state.store.history
        history.load()
        saver = asyncio.create_task(save_history_every(history, HISTORY_SAVE_S), name="history-save")
        racks = app.state.store.racks.values()
        pulled = [rack for rack in racks if rack.pull] if settings.camera_pull else []
        watched = [rack for rack in racks if rack.vision and rack.vision.url] if settings.camera_pull else []
        app.state.poll_tasks = [asyncio.create_task(CameraPoller(app, rack).run(), name=f"poll-{rack.id}") for rack in pulled]
        app.state.poll_tasks += [asyncio.create_task(VisionPoller(app, rack).run(), name=f"vision-{rack.id}") for rack in watched]
        yield
        for task in (*app.state.poll_tasks, saver):
            task.cancel()
        await asyncio.gather(*app.state.poll_tasks, saver, return_exceptions=True)
        history.save()
        await detector.aclose()
        await app.state.vision_http.aclose()
        if app.state.geocoder is not None:
            await app.state.geocoder.aclose()

    app = FastAPI(title="Find my Bike Slot", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = Store.from_yaml(settings.racks_file, settings)
    app.state.detector = detector
    app.state.detector_cache = {}
    app.state.vision_http = httpx.AsyncClient()  # previews of vision racks, fetched through the app for the admin page
    app.state.geocoder = None
    if settings.geocoder_url:
        located = [rack.location for rack in app.state.store.racks.values() if rack.location]
        app.state.geocoder = geocoder.Geocoder(settings.geocoder_url, geocoder.viewbox(located))
        geocoder.hide_queries_in_logs(settings.geocoder_url)
    app.include_router(routes_camera.router)
    app.include_router(routes_vision.router)
    app.include_router(routes_public.router)
    app.include_router(geocoder.router)
    app.include_router(routes_admin.router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def user_page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page() -> FileResponse:
        return FileResponse(STATIC / "admin.html")

    return app


app = create_app()
