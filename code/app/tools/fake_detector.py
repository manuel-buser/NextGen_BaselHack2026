"""Reference recognition service with the exact HTTP shape the app expects.

Run from code/app: uvicorn tools.fake_detector:app --port 8001, then start the app with
DETECTOR_URL=http://localhost:8001. Replace the body of analyze() with the real detector; keep the shape.
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from bikeslot.detector_client import fake_result

app = FastAPI(title="Fake bike detector")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "fake-reference", "version": "0.1.0"}


@app.post("/analyze")
async def analyze(request: Request, rack: str, capacity: int = 15) -> dict:
    image = await request.body()
    return fake_result(rack, capacity, image)
