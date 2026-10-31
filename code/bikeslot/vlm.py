"""Scene understanding with a vision-language model (OpenAI), used ONLY at calibration time.

The model is asked what kind of parking this is, to enumerate the slots one by one (free/occupied), in
which direction they are numbered and roughly where the row lies.  Pixel-precise geometry comes from the
local detector, not from here.  The uploaded frame has persons and motorcycles blurred.
Returns None whenever no key is configured or the call fails, so the pipeline keeps working offline.
"""
from __future__ import annotations

import base64
import time
from typing import Literal, Optional

from pydantic import BaseModel

from . import config
from .schemas import CalibrationHint

last_error: Optional[str] = None
last_call: Optional[dict] = None


class _Corner(BaseModel):
    x_percent: float
    y_percent: float


class _SlotState(BaseModel):
    index: int
    state: Literal["free", "occupied", "unsure"]


class _Answer(BaseModel):
    """Strict structured output: every field required, enums as Literal, no numeric constraints."""
    parking_type: Literal["painted_bays", "stirrup_rack", "wheel_holder_rack", "dense_rack", "unmarked", "unknown"]
    slots: list[_SlotState]
    slot_count: int
    row_direction: Literal["left_to_right", "right_to_left"]
    row_corners: list[_Corner]
    confidence: Literal["low", "medium", "high"]
    notes: str


_SYSTEM = (
    "You analyse a single photo of a bicycle parking area for a parking-guidance system. "
    "Be precise and conservative. Never identify or describe people."
)

_PROMPT = """Describe this bicycle parking area.

Definitions:
- A "slot" is one position where one bicycle is meant to be parked (one painted bay, one stirrup/U-rack side, one wheel holder, ...). Count ALL slots, occupied and free.
- parking_type: painted_bays (bays marked on the ground), stirrup_rack (row of metal U/stirrup stands), wheel_holder_rack (row of narrow metal wheel holders, often tilted), dense_rack (two-level or very dense racks where single slots cannot be told apart), unmarked (bikes just lean on a wall/fence), unknown.
- slots: enumerate EVERY slot of the main row one by one in numbering order (index 1, 2, 3, ...) with its state: "occupied" if a bicycle/scooter/motorbike stands in it, "free" if it is empty, "unsure" if you cannot tell. Go slowly along the row: every painted bay / stand / holder is one entry, including the ones at the far ends. slot_count must equal the number of entries.
- row_direction: the direction in which slots are numbered 1..N as seen in the image: left_to_right or right_to_left (use left_to_right unless the row clearly starts on the right).
- row_corners: the ground area that covers ALL slots of the main row, as 4 corners in percent of the image width/height, in this order: back-left, back-right, front-right, front-left ("back" = rack/wall side, "front" = aisle side closest to the camera). If several rows exist, describe the largest/closest one and mention the others in notes.
- The photo may be a picture of a screen or a printout: ignore bezels, reflections and moire and describe the parking scene shown.
Fill notes with anything useful for an admin who will verify the slot layout (e.g. "9 bays, 3 empty in the middle", "scooters parked in slots 2 and 5", "second row of racks behind")."""


def describe_scene(jpeg_bytes: bytes) -> Optional[CalibrationHint]:
    """Ask the VLM about the (blurred) frame. Never raises."""
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
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
        ]},
    ]
    models = [config.VLM_MODEL] + [m for m in ("gpt-4.1", "gpt-4o") if m != config.VLM_MODEL]
    for model in models:
        t = time.time()
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=_Answer, temperature=0, max_tokens=1500)
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
        usage = getattr(completion, "usage", None)
        last_call = {"model": model, "ms": round((time.time() - t) * 1000),
                     "tokens": getattr(usage, "total_tokens", None), "confidence": ans.confidence,
                     "slots": n, "free": free, "occupied": occupied}
        last_error = None
        states = "".join({"free": ".", "occupied": "#", "unsure": "?"}[s.state] for s in ans.slots)
        notes = (f"[{model}, confidence {ans.confidence}] {ans.notes.strip()} "
                 f"(VLM enumerates {n_enum} slots [{states}]: {occupied} occupied, {free} free)")
        return CalibrationHint(source="vlm", parking_type=ans.parking_type, slot_count=n,
                               row_direction=ans.row_direction, corners=corners, notes=notes,
                               free_count=free, occupied_count=occupied)
    return None


def info() -> dict:
    return {"configured": bool(config.OPENAI_API_KEY), "model": config.VLM_MODEL,
            "last_error": last_error, "last_call": last_call}
