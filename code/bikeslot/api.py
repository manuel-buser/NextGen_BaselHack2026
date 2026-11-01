"""bikeslot REST service.

Run from the repo root:
    .venv/Scripts/python -m uvicorn --app-dir code bikeslot.api:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs   showcase: http://localhost:8000/demo

Contract for the ESP32 team:  POST /cameras/{camera_id}/frames  with Content-Type: image/jpeg and the
JPEG bytes as body (multipart "file" also works).  Contract for the frontend: GET /sites,
GET /cameras/{id}/status, GET /cameras/{id}/preview.jpg; admin: POST .../calibrate, PUT .../layout.

Two occupancy engines share this API (per camera: ``layout.params.engine``, default BIKESLOT_ENGINE):
``detector`` (YOLO + slot geometry, occupancy.py) and ``classifier`` (fixed slot polygons + per-crop CNN,
slotcls/, see documentation/slot-classifier.md).

Hand-over to the app (code/app): the app polls GET /cameras/{id}/status itself (racks file: ``vision: {url, camera}``),
or this service pushes every new status to it when BIKESLOT_APP_URL is set (forward.py). Never pictures.

Coordinate note: ``CameraStatus.slots[].polygon`` and the layout returned with ``?pixels=true`` are pixel
coordinates of ``frame_size`` (the working frame, longer side 1280).  ``calibration-frame.jpg`` has exactly
that size, so polygons can be drawn on it 1:1; ``preview.jpg`` is downscaled to 800 px width.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from . import __version__, config, demo, detector, forward, pull, registration, vehicles, vlm
from .geometry import denormalize, normalize, resize_max_side
from .occupancy import analyze_frame
from .preview import blurred_jpeg, render
from .schemas import CameraInfo, CameraStatus, Layout, LayoutUpdate, SiteSummary, utcnow
from .store import CameraState, Store

store = Store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.SEED_GOLDEN:
        seeded = store.seed_golden()
        if seeded:
            print(f"[bikeslot] golden layout loaded for {seeded.camera_id} ({len(seeded.slots)} slots, {seeded.status})")
    if config.SEED_DEMO_LAYOUT:
        seeded = store.seed_layout_file(config.DEMO_LAYOUT)
        if seeded:
            print(f"[bikeslot] demo slot layout loaded for {seeded.camera_id} ({len(seeded.slots)} slots, engine classifier)")
        for live_id in config.DEMO_LAYOUT_CAMERAS:   # the live camera films the same hooks: same polygons, same engine
            live = store.seed_layout_file(config.DEMO_LAYOUT, camera_id=live_id)
            if live:
                print(f"[bikeslot] demo slot layout also loaded for {live.camera_id} ({len(live.slots)} slots, source {live.source})")
    print(f"[bikeslot] {len(store.cameras)} camera(s) known, loading detector {config.MODEL_NAME} ...")
    await run_in_threadpool(detector.load)
    print(f"[bikeslot] detector ready: {detector.info()}")
    # the stand detector (calibration only) takes ~30 s to load: warm it up in the background
    import threading

    from . import stands
    threading.Thread(target=stands._load, name="stands-warmup", daemon=True).start()
    pull.start(_ingest, note=store.note_source)
    forward.start()
    if config.DEMO_PREWARM:
        threading.Thread(target=demo.prewarm, name="demo-warmup", daemon=True).start()
    yield
    pull.stop_all()
    forward.stop()


app = FastAPI(title="bikeslot", version=__version__, lifespan=lifespan,
              description="Bicycle parking slot occupancy from camera snapshots (BaselHack 2026, team NextGen)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
demo.bind(store.lock)
app.include_router(demo.router)


# ----------------------------------------------------------------------------- helpers

def _cam(camera_id: str) -> CameraState:
    cam = store.get(camera_id)
    if cam is None:
        raise HTTPException(404, f"unknown camera '{camera_id}' - it is created by its first POST /cameras/{camera_id}/frames")
    return cam


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "body is not a decodable image (send JPEG/PNG bytes with Content-Type image/jpeg, or multipart 'file')")
    return img


def _engine(cam: CameraState) -> str:
    """'detector' (YOLO + slot geometry, occupancy.py) or 'classifier' (per-slot crop CNN, slotcls/).

    Chosen per camera by ``layout.params.engine`` (set by slotcls_calibrate.py), else by BIKESLOT_ENGINE."""
    return str((cam.layout.params or {}).get("engine") or config.ENGINE)


def _process(cam: CameraState, img: np.ndarray) -> CameraStatus:
    """Runs in a worker thread: analyse, render preview, record. Serialised per camera via the store lock."""
    frame, _ = resize_max_side(img, config.MAX_SIDE)
    with store.lock:
        cam.last_raw = img
        if _engine(cam) == "classifier":
            from .slotcls import pipeline as slotcls_pipeline

            try:
                res = slotcls_pipeline.analyze_frame(cam.layout, img, cam.camera_id, cam.trackers,
                                                     reference=cam.reference if config.REGISTRATION else None,
                                                     failures_before=cam.reg_failures)
            except FileNotFoundError as exc:          # no trained model yet: keep the service up, say why
                raise HTTPException(503, str(exc))
            if res.status.registration is not None:
                cam.reg_failures = 0 if res.status.registration["ok"] else cam.reg_failures + 1
        else:
            res = analyze_frame(cam.layout, cam.trackers, cam.cache, frame, cam.camera_id)
        store.push_frame(cam, res.frame)
        cam.preview_jpeg = render(res.frame, res.status, res.detections, res.assignments)
        cam.calibration_jpeg = blurred_jpeg(res.frame, res.detections, max_width=config.MAX_SIDE)
        store.record_status(cam, res.status, res.detections)
        (store.preview_dir / f"{cam.camera_id}.jpg").write_bytes(cam.preview_jpeg)
        status = store.current_status(cam)
        forward.submit(cam.camera_id, status)   # push mode: the app gets the decided slots from a background thread
        return status


def _ingest(camera_id: str, data: bytes) -> CameraStatus:
    """Pull-mode entry point (worker thread): same path as POST /cameras/{id}/frames."""
    return _process(store.get_or_create(camera_id), _decode(data))


def _layout_pixels(layout: Layout, frame_size: Optional[list[int]] = None) -> dict:
    fs = frame_size or layout.frame_size
    data = layout.model_dump(mode="json")
    for s in data["slots"]:
        s["polygon"] = [[round(x, 1), round(y, 1)] for x, y in denormalize(s["polygon"], fs)]
    for r in data["rows"]:
        r["front"] = [[round(x, 1), round(y, 1)] for x, y in denormalize(r["front"], fs)]
        r["back"] = [[round(x, 1), round(y, 1)] for x, y in denormalize(r["back"], fs)]
    data["frame_size"] = list(fs)
    data["coordinates"] = "pixels"
    return data


# ----------------------------------------------------------------------------- ingest

@app.post("/cameras/{camera_id}/frames", response_model=CameraStatus, tags=["camera"])
async def post_frame(camera_id: str, request: Request):
    """Ingest one snapshot. Body: raw JPEG (Content-Type image/jpeg) or multipart form field 'file'."""
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file") or next((v for v in form.values() if hasattr(v, "read")), None)
        if upload is None:
            raise HTTPException(400, "multipart body must contain a file field named 'file'")
        data = await upload.read()
    else:
        data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")
    img = _decode(data)
    cam = store.get_or_create(camera_id)
    return await run_in_threadpool(_process, cam, img)


@app.put("/cameras/{camera_id}/source", tags=["camera"])
def set_source(camera_id: str, url: str = Query(..., description="snapshot URL to poll, e.g. http://192.168.21.95/snapshot; empty stops polling")):
    """Let the service poll a camera's snapshot URL itself (pull mode), or stop polling with url=''."""
    old = pull.pullers.pop(camera_id, None)
    if old:
        old.stop.set()
    if url:
        pull.start(_ingest, f"{camera_id}={url}", note=store.note_source)
    return pull.info()


# ----------------------------------------------------------------------------- read

@app.get("/health", tags=["service"])
def health():
    from .slotcls import model as slotcls_model

    return {"ok": True, "version": __version__, "detector": detector.info(), "cameras": len(store.cameras),
            "vlm": vlm.info(), "pull": pull.info(), "forward": forward.info(), "engine": config.ENGINE,
            "slot_classifier": slotcls_model.info(), "registration": _registration_info(),
            "vehicle_detectors": vehicles.info(), "time": utcnow()}


def _registration_info() -> dict:
    return {"enabled": config.REGISTRATION,
            "cameras": {c.camera_id: _camera_registration(c) for c in store.cameras.values() if c.reference}}


def _camera_registration(cam: CameraState) -> dict:
    return {"reference": cam.reference.info() if cam.reference else None, "failures_in_a_row": cam.reg_failures,
            "last": cam.last_status.registration if cam.last_status else None}


@app.get("/cameras", response_model=list[CameraInfo], tags=["camera"])
def list_cameras():
    return store.camera_infos()


@app.get("/cameras/{camera_id}/status", response_model=CameraStatus, tags=["camera"])
def camera_status(camera_id: str):
    return store.current_status(_cam(camera_id))


@app.get("/cameras/{camera_id}/history", tags=["camera"])
def camera_history(camera_id: str, limit: int = Query(100, ge=1, le=5000)):
    """Last ``limit`` per-frame results (free/occupied/unknown + slot states) from the jsonl history."""
    _cam(camera_id)
    path = store.history_dir / f"{camera_id}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in lines if line.strip()]


@app.get("/cameras/{camera_id}/preview.jpg", tags=["camera"])
def camera_preview(camera_id: str):
    cam = _cam(camera_id)
    if not cam.preview_jpeg:
        raise HTTPException(404, "no frame received yet")
    return Response(cam.preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/cameras/{camera_id}/calibration-frame.jpg", tags=["admin"])
def calibration_frame(camera_id: str):
    """Blurred last frame at working resolution (matches the pixel coordinates in status/layout?pixels=true)."""
    cam = _cam(camera_id)
    if not cam.calibration_jpeg:
        raise HTTPException(404, "no frame received yet")
    return Response(cam.calibration_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/sites", response_model=list[SiteSummary], tags=["site"])
def sites():
    return store.sites()


@app.get("/sites/{site_id}", response_model=SiteSummary, tags=["site"])
def site(site_id: str):
    for s in store.sites():
        if s.site_id == site_id:
            return s
    raise HTTPException(404, f"unknown site '{site_id}'")


# ----------------------------------------------------------------------------- admin / layout

@app.get("/cameras/{camera_id}/layout", tags=["admin"])
def get_layout(camera_id: str, pixels: bool = Query(False, description="return pixel coordinates of frame_size instead of normalised 0..1")):
    cam = _cam(camera_id)
    if pixels:
        fs = cam.last_status.frame_size if cam.last_status and cam.last_status.frame_size != [0, 0] else cam.layout.frame_size
        return _layout_pixels(cam.layout, fs)
    return cam.layout


@app.put("/cameras/{camera_id}/layout", response_model=Layout, tags=["admin"])
def put_layout(camera_id: str, upd: LayoutUpdate,
               pixels: bool = Query(False, description="rows/slots in the body are pixel coordinates of frame_size")):
    """Admin saves an edited / confirmed layout. Send ``rows`` (4 corners + slot_count) and the slots are
    regenerated, or send explicit ``slots``. Set ``status`` to ``confirmed`` to enable per-slot reporting."""
    cam = store.get_or_create(camera_id)
    if pixels:
        fs = upd.frame_size or (cam.last_status.frame_size if cam.last_status and cam.last_status.frame_size != [0, 0] else cam.layout.frame_size)
        if upd.rows:
            for r in upd.rows:
                r.front = normalize(r.front, fs)
                r.back = normalize(r.back, fs)
        if upd.slots:
            for s in upd.slots:
                s.polygon = normalize(s.polygon, fs)
        upd.frame_size = list(fs)
    return store.update_layout(camera_id, upd)


@app.delete("/cameras/{camera_id}/layout", tags=["admin"])
def delete_layout(camera_id: str):
    """Forget the layout of a camera (status back to 'none'); frames keep being buffered for a new calibration."""
    cam = _cam(camera_id)
    path = store.layout_dir / f"{camera_id}.json"
    if path.exists():
        path.unlink()
    with store.lock:
        cam.layout = Layout(camera_id=camera_id, site_id=cam.layout.site_id, site_name=cam.layout.site_name)
        cam.reset_trackers()
        cam.last_status = None
    return {"ok": True}


@app.put("/cameras/{camera_id}/reference", tags=["admin"])
async def put_reference(camera_id: str, request: Request,
                        latest: bool = Query(False, description="use the newest picture the camera sent instead of a JPEG body")):
    """Store the frame the camera's layout was drawn on as its reference.  From then on every picture is warped
    onto it before the slots are cropped (registration.py), so a bumped or slightly turned camera keeps its layout.
    Send the JPEG as body (``curl --data-binary @frame.jpg``), or ``?latest=true``.  Only the features of the frame
    are kept (positions + descriptors), no pixels.  Set it together with the layout drawn on that very frame."""
    cam = store.get_or_create(camera_id)
    if latest:
        if cam.last_raw is None:
            raise HTTPException(409, "no picture received from this camera yet")
        img = cam.last_raw
    else:
        data = await request.body()
        if not data:
            raise HTTPException(400, "send the reference JPEG as body, or use ?latest=true")
        img = _decode(data)
    try:
        ref = await run_in_threadpool(registration.make_reference, img)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    store.set_reference(camera_id, ref)
    out = {"camera_id": camera_id, **ref.info()}
    lw, lh = cam.layout.frame_size
    if lw and lh and abs(lw / lh - ref.size[0] / ref.size[1]) > 0.02:
        out["warning"] = f"the layout was drawn on a {lw}x{lh} frame, the reference is {ref.size[0]}x{ref.size[1]}"
    return out


@app.get("/cameras/{camera_id}/reference", tags=["admin"])
def get_reference(camera_id: str):
    return {"camera_id": camera_id, "enabled": config.REGISTRATION, **_camera_registration(_cam(camera_id))}


@app.delete("/cameras/{camera_id}/reference", tags=["admin"])
def delete_reference(camera_id: str):
    """Stop warping this camera's pictures (they are cropped as they come again)."""
    _cam(camera_id)
    store.set_reference(camera_id, None)
    return {"ok": True}


@app.post("/cameras/{camera_id}/calibrate", tags=["admin"])
async def calibrate(camera_id: str,
                    use_vlm: bool = Query(True, description="ask the vision-language model (set-of-mark prompt) for parking type, slot count, bike->slot mapping"),
                    use_stands: bool = Query(True, description="detect rack stands with the open-vocabulary model (visual prompting)"),
                    apply: bool = Query(True, description="store the proposal as the camera's layout (status 'proposed')")):
    """Propose a slot layout from the buffered frames of this camera (Stage A). The admin then edits/confirms it
    with PUT /cameras/{camera_id}/layout."""
    cam = _cam(camera_id)
    if not cam.frames:
        raise HTTPException(409, "no frames buffered for this camera yet - POST a few snapshots first")
    from .calibrate import propose_layout  # local import: keeps the service up even if calibration deps fail

    try:
        proposal = await run_in_threadpool(propose_layout, cam, use_vlm, use_stands)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    if apply:
        proposal = store.set_layout(camera_id, proposal)
    return proposal


# ----------------------------------------------------------------------------- tiny debug dashboard

_INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>bikeslot</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
.cam{display:inline-block;margin:8px;vertical-align:top}img{max-width:640px;border:1px solid #444}
.n{font-size:28px}.free{color:#6c6}.occ{color:#e66}pre{color:#9ab}</style></head><body>
<h2>bikeslot debug view</h2><div id="sites"></div><div id="cams"></div>
<script>
async function tick(){
  const sites=await (await fetch('/sites')).json();
  document.getElementById('sites').innerHTML=sites.map(s=>`<div class="n">${s.site_name||s.site_id}: <span class="free">${s.free} free</span> / ${s.slot_count}${s.estimated?' (est.)':''}</div>`).join('');
  const cams=await (await fetch('/cameras')).json();
  document.getElementById('cams').innerHTML=cams.map(c=>`<div class="cam"><div>${c.camera_id} - layout ${c.layout_status} - ${c.online?'online':'offline'}</div>
   <img src="/cameras/${c.camera_id}/preview.jpg?t=${Date.now()}" onerror="this.style.display='none'"></div>`).join('');
}
tick(); setInterval(tick, 2000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return _INDEX
