"""Paths, thresholds and environment for the bikeslot service.

Everything tunable lives here so the demo can be adjusted without touching the pipeline code.
Values can be overridden with environment variables (prefix BIKESLOT_) or a .env file at the repo root.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env(path: Path) -> None:
    """Load KEY=value lines from .env (python-dotenv if installed, else a minimal parser)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")


def _env(name: str, default):
    raw = os.getenv(f"BIKESLOT_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# --- paths -------------------------------------------------------------------
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
LAYOUT_DIR = DATA_DIR / "layouts"
HISTORY_DIR = DATA_DIR / "history"
PREVIEW_DIR = DATA_DIR / "preview"
GOLDEN_DIR = ROOT / "assets" / "pictures" / "GoldenDataSet"
GOLDEN_LABELS = GOLDEN_DIR / "golden_labels.json"

# --- detector ----------------------------------------------------------------
# yolo11x-seg gives masks -> true ground-contact point of each wheel; yolo11x (det only) also works.
MODEL_NAME = _env("MODEL", "yolo11x-seg.pt")
MODEL_PATH = ROOT / MODEL_NAME
DEVICE = _env("DEVICE", "0")            # "0" = first GPU, "cpu" to force CPU
IMGSZ = _env("IMGSZ", 1280)
MAX_SIDE = _env("MAX_SIDE", 1280)       # frames are resized so their longer side equals this
CONF = _env("CONF", 0.2)
IOU = _env("IOU", 0.5)
HALF = _env("HALF", True)
# COCO class ids
CLS_PERSON, CLS_BICYCLE, CLS_MOTORCYCLE = 0, 1, 3
VEHICLE_CLASSES = {CLS_BICYCLE: "bicycle", CLS_MOTORCYCLE: "motorcycle"}
CLASS_NAMES = {CLS_PERSON: "person", **VEHICLE_CLASSES}

# --- occupancy ---------------------------------------------------------------
FOOTPRINT_FRAC = _env("FOOTPRINT_FRAC", 0.30)   # bottom share of a box/mask treated as "on the ground"
CONTACT_FRAC = _env("CONTACT_FRAC", 0.06)       # lowest share of a mask used as the wheel contact point
MIN_ASSIGN_SCORE = _env("MIN_ASSIGN_SCORE", 0.3)
K_FRAMES = _env("K_FRAMES", 5)                  # smoothing window
OCCUPIED_ABOVE = _env("OCCUPIED_ABOVE", 0.6)
FREE_BELOW = _env("FREE_BELOW", 0.4)
PERSON_COVER_UNKNOWN = _env("PERSON_COVER_UNKNOWN", 0.5)
BLUR_VAR_MIN = _env("BLUR_VAR_MIN", 30.0)       # Laplacian variance below this -> frame too blurry
OFFLINE_AFTER_S = _env("OFFLINE_AFTER_S", 60.0)
RING_BUFFER = _env("RING_BUFFER", 10)           # raw frames kept in RAM per camera (calibration)

# --- calibration -------------------------------------------------------------
STAND_MODEL = _env("STAND_MODEL", "yoloe-11l-seg.pt")   # open-vocabulary model for rack stands (calibration only)
STAND_CONF = _env("STAND_CONF", 0.3)                    # visual-prompt confidence to accept a stand
STAND_EXEMPLAR_MIN_CONF = _env("STAND_EXEMPLAR_MIN_CONF", 0.15)  # text-prompt confidence needed for the exemplar
SLOT_DEPTH_PER_STAND_HEIGHT = _env("SLOT_DEPTH_PER_STAND_HEIGHT", 3.0)  # slot depth when no bike gives the front line
CLUSTER_GAP_FRAC = _env("CLUSTER_GAP_FRAC", 0.5)   # new slot cluster when gap > frac * median box width
MAX_SLOTS = _env("MAX_SLOTS", 60)

# --- VLM ---------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("MY_API_KEY")  # .env may use either name
VLM_MODEL = _env("VLM_MODEL", "gpt-4.1")
VLM_MAX_SIDE = _env("VLM_MAX_SIDE", 1024)

# --- preview / privacy -------------------------------------------------------
PREVIEW_MAX_WIDTH = _env("PREVIEW_MAX_WIDTH", 800)
BLUR_CLASSES = {CLS_PERSON, CLS_MOTORCYCLE}      # faces + licence plates are never shown or uploaded

# --- service -----------------------------------------------------------------
SEED_GOLDEN = _env("SEED_GOLDEN", True)
DEMO_PREWARM = _env("DEMO_PREWARM", True)   # compute the /demo example images in the background at start

# --- pull mode ---------------------------------------------------------------
# cameras the service polls itself: "cam_id=url,cam_id2=url2" (empty string disables pulling)
PULL_CAMERAS = _env("PULL_CAMERAS", "live01=http://192.168.21.95/snapshot")
PULL_INTERVAL = _env("PULL_INTERVAL", 5.0)     # seconds between snapshots per camera
PULL_TIMEOUT = _env("PULL_TIMEOUT", 10.0)

# --- slot classifier (second approach: fixed slot polygons + per-crop CNN, see documentation/slot-classifier.md)
ENGINE = _env("ENGINE", "detector")      # "detector" = YOLO + geometry (occupancy.py); "classifier" = slotcls per-crop CNN
SLOTCLS_MODEL = Path(_env("SLOTCLS_MODEL", str(ROOT / "assets" / "models" / "slotcls_demo.pth")))
SLOTCLS_BACKBONE = _env("SLOTCLS_BACKBONE", "mobilenet_v3_small")   # torchvision ImageNet backbone
SLOTCLS_INPUT = _env("SLOTCLS_INPUT", 96)          # crop side in pixels fed to the classifier
SLOTCLS_MARGIN = _env("SLOTCLS_MARGIN", 0.0)       # grow each slot polygon by this fraction before cropping (0 was best on the demo frame)
SLOTCLS_K_FRAMES = _env("SLOTCLS_K_FRAMES", 5)     # majority vote window (hysteresis: OCCUPIED_ABOVE / FREE_BELOW)
SLOTCLS_MIN_FRAMES = _env("SLOTCLS_MIN_FRAMES", 2) # votes needed before a slot leaves "unknown"
SLOTCLS_UNSURE_BAND = _env("SLOTCLS_UNSURE_BAND", 0.1)   # |p - 0.5| below this -> no vote in this frame
SLOTCLS_BLUR_PERSONS = _env("SLOTCLS_BLUR_PERSONS", True)  # run the YOLO detector only to blur persons in previews
DEMO_LAYOUT = ROOT / "assets" / "pictures" / "DemoPictures" / "demo_picture.layout.json"   # 19 hooks, camera demo01
SEED_DEMO_LAYOUT = _env("SEED_DEMO_LAYOUT", True)   # load DEMO_LAYOUT for camera demo01 at start (classifier engine)
# the live camera films the same 19 hooks the demo polygons were drawn on: seed them for these camera ids too
# (slot ids renamed, classifier engine); a layout saved under data/layouts/<id>.json wins. "" = only demo01.
DEMO_LAYOUT_CAMERAS = [c.strip() for c in _env("DEMO_LAYOUT_CAMERAS", "live01").split(",") if c.strip()]
DEMO_LABELS = DEMO_LAYOUT.with_name("demo_picture.labels.json")   # hand labels of the demo picture (for /demo)

# --- app (push mode, forward.py) ------------------------------------------
# The app (code/app) normally polls GET /cameras/{id}/status itself. Set APP_URL to push every new status to it
# instead (an app on a server that cannot reach this laptop). APP_RACKS maps camera id -> rack id in the app.
APP_URL = _env("APP_URL", "")                 # e.g. http://192.168.21.50:8080 or https://bikeslot.example.org
APP_RACKS = _env("APP_RACKS", "live01=MX-CAM")  # "camera=rack,camera2=rack2"
APP_TOKEN = _env("APP_TOKEN", "")             # the app's VISION_TOKEN, sent as X-Vision-Token
APP_TIMEOUT = _env("APP_TIMEOUT", 5.0)
