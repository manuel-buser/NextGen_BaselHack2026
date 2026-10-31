"""Slot-classifier approach (alternative to the detector + geometry pipeline in ``bikeslot.occupancy``).

    calibrate once   : one polygon per slot, drawn on the camera image, stored as the camera's Layout
    per frame        : crop every slot polygon (perspective-rectified), resize to ~96x96, binary CNN free/occupied
    temporal         : majority vote over the last N frames + hysteresis

Nothing here detects bikes or counts slots: with a fixed camera the slots never move, so only *what is in
them* has to be decided.  See documentation/slot-classifier.md.
"""
