"""Pydantic models shared by the pipeline, the REST API and the scripts.

Coordinate conventions
----------------------
* Layout polygons/rows are stored NORMALISED (0..1) relative to ``frame_size`` so that the same layout
  works for 1600x1200 stills and 800x600 streams.  The API additionally returns pixel polygons.
* Polygon vertex order for a slot: back-left, back-right, front-right, front-left (as in golden_labels.json).
* Detections are in pixel coordinates of the working frame (longer side == config.MAX_SIDE).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Point = list[float]  # [x, y]

ParkingType = Literal["painted_bays", "stirrup_rack", "wheel_holder_rack", "dense_rack", "unmarked", "unknown"]
LayoutStatus = Literal["none", "proposed", "confirmed"]
OccupancyMode = Literal["per_slot", "count_only"]
RowDirection = Literal["left_to_right", "right_to_left"]
SlotState = Literal["free", "occupied", "unknown"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- layout
class Row(BaseModel):
    """A straight row of equally spaced slots between a front line and a back line (normalised coords)."""
    row_id: str = "r1"
    slot_count: int = Field(ge=1)
    front: list[Point] = Field(description="[P0, P1]: ground line closest to the camera / aisle, left->right")
    back: list[Point] = Field(description="[Q0, Q1]: ground line at the rack / wall side, left->right")
    direction: RowDirection = "left_to_right"


class Slot(BaseModel):
    slot_id: str
    index: int = Field(ge=1, description="1..N in row direction")
    row_id: Optional[str] = None
    polygon: list[Point] = Field(description="normalised 0..1, back-left, back-right, front-right, front-left")


class CalibrationHint(BaseModel):
    """What the scene-understanding step believes about the parking area (VLM or local heuristics)."""
    source: Literal["vlm", "local", "none"] = "none"
    parking_type: ParkingType = "unknown"
    slot_count: int = 0
    row_direction: RowDirection = "left_to_right"
    corners: Optional[list[Point]] = Field(
        default=None, description="normalised back-left, back-right, front-right, front-left of the whole row")
    free_count: Optional[int] = None
    occupied_count: Optional[int] = None
    notes: str = ""


class Layout(BaseModel):
    camera_id: str
    site_id: str = "site01"
    site_name: str = ""
    frame_size: list[int] = Field(default=[1280, 960], description="[width, height] the layout was drawn on")
    parking_type: ParkingType = "unknown"
    status: LayoutStatus = "none"
    source: Literal["golden", "auto", "admin", "none"] = "none"
    occupancy_mode: OccupancyMode = "per_slot"
    rows: list[Row] = []
    slots: list[Slot] = []
    calibration_hint: Optional[CalibrationHint] = None
    params: dict = {}
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def slot_count(self) -> int:
        return len(self.slots)


class LayoutUpdate(BaseModel):
    """PUT body: every field optional; rows without slots -> slots are regenerated from the rows."""
    site_id: Optional[str] = None
    site_name: Optional[str] = None
    frame_size: Optional[list[int]] = None
    parking_type: Optional[ParkingType] = None
    status: Optional[LayoutStatus] = None
    occupancy_mode: Optional[OccupancyMode] = None
    rows: Optional[list[Row]] = None
    slots: Optional[list[Slot]] = None
    params: Optional[dict] = None


# --------------------------------------------------------------------------- detection
class Detection(BaseModel):
    cls: int
    name: str
    conf: float
    box: list[float] = Field(description="[x1, y1, x2, y2] pixels in the working frame")
    contact: Optional[Point] = Field(default=None, description="ground contact point (lowest mask pixels or bottom-centre)")
    mask: Optional[list[Point]] = Field(default=None, description="mask outline polygon, pixels")

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def bottom_center(self) -> Point:
        return [(self.box[0] + self.box[2]) / 2.0, self.box[3]]


# --------------------------------------------------------------------------- status
class SlotStatus(BaseModel):
    slot_id: str
    index: int
    status: SlotState = "unknown"
    confidence: float = 0.0
    score: float = 0.0
    last_change: Optional[datetime] = None
    polygon: list[Point] = Field(default=[], description="pixel coords for the frame_size in CameraStatus")


class CameraStatus(BaseModel):
    camera_id: str
    site_id: str = "site01"
    online: bool = False
    frame_age_s: Optional[float] = None
    updated_at: Optional[datetime] = None
    layout_status: LayoutStatus = "none"
    occupancy_mode: OccupancyMode = "per_slot"
    estimated: bool = False
    frame_size: list[int] = [0, 0]
    slot_count: int = 0
    free: int = 0
    occupied: int = 0
    unknown: int = 0
    slots: list[SlotStatus] = []
    detections: dict[str, int] = {}
    processing_ms: float = 0.0
    preview_url: Optional[str] = None
    message: str = ""


class CameraInfo(BaseModel):
    camera_id: str
    site_id: str
    site_name: str = ""
    layout_status: LayoutStatus
    occupancy_mode: OccupancyMode
    slot_count: int
    online: bool
    last_frame_at: Optional[datetime] = None
    frames_buffered: int = 0


class SiteCamera(BaseModel):
    camera_id: str
    slot_count: int
    free: int
    occupied: int
    unknown: int
    online: bool
    layout_status: LayoutStatus
    estimated: bool = False


class SiteSummary(BaseModel):
    site_id: str
    site_name: str = ""
    slot_count: int = 0
    free: int = 0
    occupied: int = 0
    unknown: int = 0
    estimated: bool = False
    free_slot_ids: list[str] = []
    cameras: list[SiteCamera] = []
    updated_at: Optional[datetime] = None
