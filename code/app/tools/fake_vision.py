"""Stand-in for the vision service (code/bikeslot): serves the same JSON the real service serves, without a camera
or a GPU, so the app's vision path can be developed and rehearsed on any laptop.

Run from code/app: uvicorn tools.fake_vision:app --port 8020, then give a rack in the racks file
vision: {url: "http://127.0.0.1:8020", camera: live01} and start the app (racks.memox.yaml has that rack).
Endpoints as in the real service: GET /health, GET /cameras/{id}/status, GET /cameras/{id}/preview.jpg (an SVG
drawing of the hooks here; the real one is the blurred camera picture). A new "frame" is decided every FRAME_S
seconds. Rehearse the live demo by hand:
    curl -X POST http://127.0.0.1:8020/fake/park/7     # a bike arrives at hook 7
    curl -X POST http://127.0.0.1:8020/fake/leave/7    # it leaves again
    curl -X POST "http://127.0.0.1:8020/fake/set?occupied=3,6,10,14,18"
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response

CAPACITY = 19
CAMERA = "live01"
OCCUPIED = (4, 7, 11, 15, 19)  # the demo picture: 5 bikes on the 19 hooks of bay 2
FRAME_S = 5.0


def make_app(capacity: int = CAPACITY, occupied=OCCUPIED, camera: str = CAMERA, frame_s: float = FRAME_S, drift: bool = False) -> FastAPI:
    """A fresh service with one camera. frame_s=0 decides a new frame on every request (tests); drift makes a
    random bike arrive or leave now and then (a livelier rehearsal)."""
    app = FastAPI(title="Fake vision service")
    state = {"occupied": set(occupied), "frames": 0, "started": time.time(), "frame_at": 0.0, "updated_at": None}

    def frame() -> None:
        """Advance to a new decided frame when the last one is old enough."""
        now = time.time()
        if state["updated_at"] is not None and now - state["frame_at"] < frame_s:
            return
        if drift and state["frames"] and random.random() < 0.3:
            hook = random.randint(1, capacity)
            state["occupied"].symmetric_difference_update({hook})
        state["frames"] += 1
        state["frame_at"] = now
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

    def status() -> dict:
        frame()
        slots = [
            {"slot_id": f"{camera}-S{i:02d}", "index": i, "status": "occupied" if i in state["occupied"] else "free",
             "confidence": 1.0, "score": 0.97 if i in state["occupied"] else 0.03, "last_change": None, "polygon": []}
            for i in range(1, capacity + 1)
        ]
        taken = len(state["occupied"])
        return {
            "camera_id": camera, "site_id": "site01", "online": True, "frame_age_s": round(time.time() - state["frame_at"], 1),
            "updated_at": state["updated_at"], "layout_status": "confirmed", "occupancy_mode": "per_slot", "estimated": False,
            "frame_size": [1280, 960], "slot_count": capacity, "free": capacity - taken, "occupied": taken, "unknown": 0,
            "slots": slots, "detections": {"person": 0, "bicycle": taken, "motorcycle": 0}, "processing_ms": 12.5,
            "preview_url": f"/cameras/{camera}/preview.jpg", "message": "",
        }

    def known(camera_id: str) -> None:
        if camera_id != camera:
            raise HTTPException(status_code=404, detail=f"unknown camera '{camera_id}' - it is created by its first POST /cameras/{camera_id}/frames")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "version": "fake", "detector": {"loaded": False, "model": "fake-vision"}, "cameras": 1,
                "engine": "classifier", "slot_classifier": {"loaded": True, "backbone": "fake"}, "time": datetime.now(timezone.utc).isoformat()}

    @app.get("/cameras")
    async def cameras() -> list[dict]:
        return [{"camera_id": camera, "site_id": "site01", "site_name": "Fake wall rack", "layout_status": "confirmed",
                 "occupancy_mode": "per_slot", "slot_count": capacity, "online": True, "last_frame_at": state["updated_at"], "frames_buffered": 0}]

    @app.get("/cameras/{camera_id}/status")
    async def camera_status(camera_id: str) -> dict:
        known(camera_id)
        return status()

    @app.get("/cameras/{camera_id}/preview.jpg")
    async def camera_preview(camera_id: str) -> Response:
        known(camera_id)
        frame()
        cell = 40
        hooks = "".join(
            f'<rect x="{10 + (i - 1) * cell}" y="30" width="{cell - 4}" height="60" rx="4" fill="{"#d9363e" if i in state["occupied"] else "#2f9e44"}"/>'
            f'<text x="{10 + (i - 1) * cell + cell / 2 - 2}" y="66" font-family="sans-serif" font-size="14" fill="#fff" text-anchor="middle">{i}</text>'
            for i in range(1, capacity + 1)
        )
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {20 + capacity * cell} 110"><rect width="100%" height="100%" fill="#222"/>'
               f'<text x="10" y="18" font-family="sans-serif" font-size="13" fill="#ccc">fake vision service, {camera}, {capacity - len(state["occupied"])} free of {capacity}</text>{hooks}</svg>')
        return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.post("/fake/park/{index}")
    async def park(index: int) -> dict:
        state["occupied"].add(index)
        state["frame_at"] = 0.0  # the next status is a new frame
        return {"occupied": sorted(state["occupied"])}

    @app.post("/fake/leave/{index}")
    async def leave(index: int) -> dict:
        state["occupied"].discard(index)
        state["frame_at"] = 0.0
        return {"occupied": sorted(state["occupied"])}

    @app.post("/fake/set")
    async def set_occupied(occupied: str = "") -> dict:
        state["occupied"] = {int(part) for part in occupied.split(",") if part.strip()}
        state["frame_at"] = 0.0
        return {"occupied": sorted(state["occupied"])}

    return app


app = make_app(drift=True)
