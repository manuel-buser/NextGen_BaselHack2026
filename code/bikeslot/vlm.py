"""Scene understanding with a vision-language model (OpenAI), used ONLY at calibration time.

Set-of-mark prompting: the frame is sent with numbered boxes drawn on the detected vehicles (and, when
available, on detected stands).  The model is asked to enumerate every slot of the row one by one, to say
which slot index each marked vehicle stands in, the numbering direction, the parking type and rough row
corners.  Index-to-pixel pairs are far more reliable than a bare count and let the caller fit the slot
spacing.  Persons and motorcycles are blurred before upload.  Returns None whenever no key is configured or
the call fails, so the pipeline keeps working offline.
"""
from __future__ import annotations

import base64
import time
from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from . import config
from .schemas import CalibrationHint, Detection

last_error: Optional[str] = None
last_call: Optional[dict] = None


class _Corner(BaseModel):
    x_percent: float
    y_percent: float


class _SlotState(BaseModel):
    index: int
    state: Literal["free", "occupied", "unsure"]


class _MarkAssignment(BaseModel):
    mark: int
    slot_index: int          # 0 = not in any slot of the main row


class _Answer(BaseModel):
    """Strict structured output: every field required, enums as Literal, no numeric constraints."""
    parking_type: Literal["painted_bays", "stirrup_rack", "wheel_holder_rack", "dense_rack", "unmarked", "unknown"]
    slots: list[_SlotState]
    slot_count: int
    marked_vehicles: list[_MarkAssignment]
    row_direction: Literal["left_to_right", "right_to_left"]
    row_corners: list[_Corner]
    confidence: Literal["low", "medium", "high"]
    notes: str


_SYSTEM = (
    "You analyse a single photo of a bicycle parking area for a parking-guidance system. "
    "Be precise and conservative. Never identify or describe people."
)

_PROMPT = """Describe this bicycle parking area. Vehicles that our detector found are marked with numbered boxes ("1", "2", ...).

Definitions:
- A "slot" is one position where one bicycle is meant to be parked (one painted bay, one stirrup/U-rack side, one wheel holder, ...). Count ALL slots, occupied and free.
- parking_type: painted_bays (bays marked on the ground), stirrup_rack (row of metal U/stirrup stands), wheel_holder_rack (row of narrow metal wheel holders, often tilted), dense_rack (two-level or very dense racks where single slots cannot be told apart), unmarked (bikes just lean on a wall/fence), unknown.
- slots: enumerate EVERY slot of the main row one by one in numbering order (index 1, 2, 3, ...) with its state: "occupied" if a bicycle/scooter/motorbike stands in it, "free" if it is empty, "unsure" if you cannot tell. Go slowly along the row: every painted bay / stand / holder is one entry, including the ones at the far ends and the ones partly hidden behind bikes. slot_count must equal the number of entries.
- marked_vehicles: for every numbered box, the slot index (1..N) the vehicle stands in, or 0 if it is not in the main row. Two vehicles cannot share a slot unless the slot is clearly oversized.
- row_direction: the direction in which slots are numbered 1..N as seen in the image: left_to_right or right_to_left (use left_to_right unless the row clearly starts on the right).
- row_corners: the ground area that covers ALL slots of the main row, as 4 corners in percent of the image width/height, in this order: back-left, back-right, front-right, front-left ("back" = rack/wall side, "front" = aisle side closest to the camera). Follow the real shape of the row (it may be a skewed parallelogram, not a rectangle). If several rows exist, describe the largest/closest one and mention the others in notes.
- The photo may be a picture of a screen or a printout: ignore bezels, reflections and moire and describe the parking scene shown.
Fill notes with anything useful for an admin who will verify the slot layout (e.g. "9 bays, 3 empty in the middle", "scooter in slot 5", "second row behind")."""


def mark_frame(frame: np.ndarray, dets: list[Detection], marks: list[int]) -> np.ndarray:
    """Draw big numbered boxes on the given detections (index into ``dets``) - set-of-mark prompting."""
    img = frame.copy()
    for k, i in enumerate(marks, start=1):
        d = dets[i]
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        label = str(k)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(img, (cx - tw // 2 - 8, cy - th // 2 - 8), (cx + tw // 2 + 8, cy + th // 2 + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    return img


def describe_scene(jpeg_bytes: bytes, n_marks: int = 0) -> Optional[CalibrationHint]:
    """Ask the VLM about the (blurred, marked) frame. Never raises."""
    global last_error, last_call
    if not config.OPENAI_API_KEY:
        last_error = "no OPENAI_API_KEY / MY_API_KEY configured"
        return None
    try:
        from openai import OpenAI
    except ImportError:
        last_error = "openai package not installed"
        return None
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = _PROMPT + (f"\nThere are {n_marks} numbered boxes (1..{n_marks})." if n_marks else
                        "\nNo vehicles were marked; marked_vehicles must be an empty list.")
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
        ]},
    ]
    models = [config.VLM_MODEL] + [m for m in ("gpt-4.1", "gpt-4o") if m != config.VLM_MODEL]
    for model in models:
        t = time.time()
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=_Answer, temperature=0, max_tokens=1800)
        except Exception as exc:  # unknown model, network, auth ... try the next model
            last_error = f"{model}: {type(exc).__name__}: {str(exc)[:200]}"
            continue
        msg = completion.choices[0].message
        if getattr(msg, "refusal", None) or msg.parsed is None:
            last_error = f"{model}: refusal or empty answer"
            continue
        ans: _Answer = msg.parsed
        corners = None
        if len(ans.row_corners) == 4:
            corners = [[min(1.0, max(0.0, c.x_percent / 100.0)), min(1.0, max(0.0, c.y_percent / 100.0))]
                       for c in ans.row_corners]
        n_enum = len(ans.slots)
        n = max(n_enum, ans.slot_count, 0)
        free = sum(1 for s in ans.slots if s.state == "free")
        occupied = sum(1 for s in ans.slots if s.state == "occupied")
        mark_slots = {m.mark: m.slot_index for m in ans.marked_vehicles if 1 <= m.mark <= n_marks and m.slot_index >= 0}
        usage = getattr(completion, "usage", None)
        last_call = {"model": model, "ms": round((time.time() - t) * 1000),
                     "tokens": getattr(usage, "total_tokens", None), "confidence": ans.confidence,
                     "slots": n, "free": free, "occupied": occupied, "marks": mark_slots}
        last_error = None
        states = "".join({"free": ".", "occupied": "#", "unsure": "?"}[s.state] for s in ans.slots)
        notes = (f"[{model}, confidence {ans.confidence}] {ans.notes.strip()} "
                 f"(VLM enumerates {n_enum} slots [{states}]: {occupied} occupied, {free} free; "
                 f"marked vehicles -> slots {mark_slots})")
        return CalibrationHint(source="vlm", parking_type=ans.parking_type, slot_count=n,
                               row_direction=ans.row_direction, corners=corners, notes=notes,
                               free_count=free, occupied_count=occupied, mark_slots=mark_slots)
    return None


def info() -> dict:
    return {"configured": bool(config.OPENAI_API_KEY), "model": config.VLM_MODEL,
            "last_error": last_error, "last_call": last_call}
