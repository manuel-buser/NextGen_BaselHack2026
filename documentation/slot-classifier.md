# Slot classifier: fixed polygons + binary free/occupied CNN (`code/bikeslot/slotcls`)

Second occupancy approach, added 2026-10-31 next to the detector pipeline in
[vision-pipeline.md](vision-pipeline.md). Nothing is deleted: both engines serve the same REST API and the
same `CameraStatus` JSON, the engine is chosen per camera.

## Idea

The camera is fixed, so the slots never move. Instead of finding bikes and counting slots in every frame:

1. **Calibrate once.** One polygon per slot, drawn on the camera image, stored as the camera's layout
   (same `Layout` JSON as before: normalised 0..1 coordinates, vertex order back-left, back-right,
   front-right, front-left). For a straight rack with equal spacing this is 4 corners + the slot count,
   split in perspective.
2. **Per frame** (one JPEG every few seconds, no video): every polygon is warped to a 96x96 patch and a
   small ImageNet-pretrained CNN (MobileNetV3-small, 2.5 M parameters) answers *free or occupied*.
   This is the standard recipe of the car-parking literature (PKLot, CNRPark) and runs on a Raspberry
   Pi or Jetson.
3. **Temporal smoothing.** Majority vote over the last 5 frames with hysteresis (occupied above 0.6,
   free below 0.4 of the votes), a probability within 0.1 of 0.5 casts no vote. A person walking through
   or a bike half-way into the hook does not flip the state.

Labelling is one click per slot per frame, so a few hundred labelled frames are an hour of work.

## The demo picture

`assets/pictures/DemoPictures/demo_picture.jpeg` (2048x1536): the wall rack in front of Memox, Elisabethenstrasse 15,
seen by the ESP32 camera by day. The rack has four bays between concrete pillars, counted on site on 2026-10-31:
10 hooks, then 19, 19 and 19. The camera films bay 2: its **19** hooks are the slots, numbered left to right; the
hooks of the other bays are ignored. The hook pitch is close to constant in the image (about 63 px), so the 19-way
split of the row quadrilateral matches the hooks. (The first calibration on 2026-10-31 counted 18: hook 3 is
hidden behind the bike hanging on hook 4 from the camera's position. It is labelled `unknown` on that picture and
neither trained nor scored; from that angle the count is one low whenever hook 4 is taken.)

| File | What |
|---|---|
| `demo_picture.layout.json` | the 19 slot polygons (camera `demo01`, `params.engine = "classifier"`); loaded by the service at start for `demo01` and for the live camera `live01` |
| `demo_picture.labels.json` | ground truth: slots 4, 7, 11, 15, 19 occupied (bikes 1, 4 and 5 hang in the hook, bikes 2 and 3 lean), 13 free, slot 3 unknown (hidden) |
| `memox_01.jpeg` ... `memox_10.jpeg` | phone pictures of bays 2, 3 and 4 taken on site on the evening of 2026-10-31 (night light, head-on), each with its own `.layout.json` (found by `slotcls_hooks.py`), `.labels.json` (by eye), `.overlay.jpg`, `.crops.jpg` |
| `demo_picture.overlay.jpg` / `.crops.jpg` | the polygons on the image and the 18 crops the classifier sees |
| `assets/models/slotcls_demo.pth` | the trained head (4 KB; the backbone weights come from torchvision) |

A slot counts as occupied when a bike's front wheel is at its hook or the bike leans on that hook. A
handlebar reaching into the neighbouring slot does not occupy it (the hook is still usable).

### Numbers (11 labelled frames, 208 crops, state of 2026-10-31 evening)

| Check | Result |
|---|---|
| Leave-one-frame-out over the 11 frames (train on 10, predict the 11th), MobileNetV3-small 96x96, no margin | **192 / 208 (92 %)** |
| of which the ten night phone pictures (head-on, bays 2 to 4) | 180 / 190 (95 %) |
| of which the camera's own day picture (angled), held out alone | 12 / 18: the only frame of that viewpoint and light in the set |
| Final model on its own training frames (sanity check only) | 194 / 208 |
| Before, with the single frame: leave-one-slot-out on the 18-slot layout | 15 / 18 |
| Classifier per frame, 19 crops, laptop GPU | ~10 ms (+ ~80 ms when the person-blur detector pass is on) |

What the errors are (`slotcls_train.py --cv frames` lists them): on the phone pictures, 8 of the 10 are a free hook
whose column is half filled by the neighbour's wheel or basket (the model says occupied, p 0.5 to 0.9), 2 are a bike
leaning at a flat angle that the model takes for free. On the camera's day picture held out, the free hooks behind
the leaning bikes are called occupied: the model has never seen that angle without this frame. With a single frame
of that view in the training set (the final model) it gets that frame right again, but the honest expectation for
the mounted camera tomorrow is "about one hook in ten wrong, on the free hooks next to leaning bikes" until a few
frames of the mounted camera itself (daylight, its angle, different bikes) are labelled and the head retrained:
that is one click per changed hook in `slotcls_label.py` and ten seconds of training. The old pipeline's detector is
not used for the decision.

The shipped `assets/models/slotcls_demo.pth` is trained on all eleven frames with the camera's own picture repeated
six times (its label file given six times to `slotcls_train.py`), so the angled view weighs as much as the night
pictures: on its training frames it gets the camera picture 14 / 18 (12 / 18 with equal weights; the remaining
misses are the free hooks 6, 10, 14, 18 to the left of a leaning bike, whose column the bike fills from that angle)
and the night pictures 179 / 190 (181 / 190 with equal weights). Leave-one-frame-out above is with equal weights.
Note the two labelling conventions in the set: the night pictures were labelled "a bike body filling the column
counts as occupied", the camera picture by hook ("a leaning neighbour does not occupy the hook"); settle on one
before labelling frames of the mounted camera.

## How to use it

```bash
# 1. calibrate once (4 corners + N; or --interactive to click them; or edit polygons in the JSON / admin UI)
.venv/Scripts/python code/scripts/slotcls_calibrate.py --image assets/pictures/DemoPictures/demo_picture.jpeg \
    --camera demo01 --slots 18 --corners 248,770 1462,910 1503,1165 112,1025 \
    --out assets/pictures/DemoPictures/demo_picture.layout.json
#    writes demo_picture.overlay.jpg (check the numbering) and demo_picture.crops.jpg (check the crops)

# 2. label frames: click a slot to toggle free/occupied, Enter = next frame; --model pre-fills the labels
.venv/Scripts/python code/scripts/slotcls_label.py --layout assets/pictures/DemoPictures/demo_picture.layout.json \
    --images data/snapshots/demo01 --out data/slotcls/demo01.labels.json --model assets/models/slotcls_demo.pth

# 3. train (head only; --finetune once there are a few thousand crops); --cv gives an honest number first
.venv/Scripts/python code/scripts/slotcls_train.py --labels assets/pictures/DemoPictures/demo_picture.labels.json \
    assets/pictures/DemoPictures/memox_*.labels.json data/slotcls/demo01.labels.json --out assets/models/slotcls_demo.pth --cv frames
#    (a label file given several times weighs its frames up: the shipped model has the camera frame six times)

# 4. run on images / a folder (sequence with smoothing), overlays go to data/slotcls/runs
.venv/Scripts/python code/scripts/slotcls_run.py --layout assets/pictures/DemoPictures/demo_picture.layout.json \
    --images assets/pictures/DemoPictures/demo_picture.jpeg --labels assets/pictures/DemoPictures/demo_picture.labels.json

# 5. service: cameras demo01 and live01 (BIKESLOT_DEMO_LAYOUT_CAMERAS) are seeded with the demo layout and use the
#    classifier automatically; the real camera at the demo spot therefore reports the 18 hooks from its first frames.
#    To redraw the polygons for the live camera: slotcls_calibrate.py --camera live01 --out data/layouts/live01.json,
#    then PUT the file to /cameras/live01/layout (or restart). The app reads the result: documentation/demo-runbook.md
.venv/Scripts/python -m uvicorn --app-dir code bikeslot.api:app --host 0.0.0.0 --port 8000
.venv/Scripts/python code/scripts/replay.py --camera demo01 --source assets/pictures/DemoPictures/demo_picture.jpeg --interval 3 --loop
```

Engine selection: `layout.params.engine` (`"classifier"` is set by `slotcls_calibrate.py`), otherwise
`BIKESLOT_ENGINE` (default `detector`). `GET /health` shows `engine` and `slot_classifier`. A camera whose
layout was drawn in the admin UI (`PUT /cameras/{id}/layout`) uses the classifier when its `params` contain
`{"engine": "classifier"}`.

Tunables in `config.py`: `SLOTCLS_MODEL` (default `assets/models/slotcls_demo.pth`), `SLOTCLS_BACKBONE`,
`SLOTCLS_INPUT` (96 or `64x160`), `SLOTCLS_MARGIN`, `SLOTCLS_K_FRAMES`, `SLOTCLS_MIN_FRAMES`,
`SLOTCLS_UNSURE_BAND`, `SLOTCLS_BLUR_PERSONS` (run YOLO only to blur persons in previews and to mark slots
hidden behind a person as unknown; switch off on a device without GPU).

## Camera bumps: warping every picture onto the reference frame (`registration.py`)

The polygons are pixel positions, so a camera that is bumped, turned or tilted needs a new layout - unless the
layout's **reference frame** (the picture the polygons were drawn on) is stored too. Then every picture is
warped onto that frame before the slots are cropped:

1. SIFT keypoints + descriptors, found at 320 px width (`REG_WORK_WIDTH`, half the camera's 640 x 480) after a
   CLAHE contrast boost, with burnt-out pixels (lamps) masked: only coarse structure (pillars, rail, graffiti,
   windows) is used, never pixel noise or JPEG blocks, and a 320 x 240 picture registers just as well.
2. Matches with the Lowe ratio test (0.8), then a RANSAC homography (similarity transform as fallback: rotation +
   uniform scale + shift).
3. Plausibility: at least 15 agreeing matches, scale 0.6-1.6, rotation <= 25 deg, little perspective change.
4. `cv2.warpPerspective` onto the reference frame; the fixed polygons fit. A slot the camera no longer sees
   (less than 90 % inside the warped picture) gets no vote.

A picture that cannot be warped (camera turned away, covered, picked up) casts no votes; after three in a row
(`REG_FAIL_RESET`) every slot is unknown (the app counts it as taken) and the status says so. The preview banner
shows `aligned` or `NOT ALIGNED`; `CameraStatus.registration` and `GET /health` carry matches, scale, rotation,
shift and the time taken (~15-20 ms per picture on a laptop CPU).

Measured on the night frames of 2026-11-01 (five camera positions, up to 11 deg turned and 100 px shifted): hooks
mapped within 0.4-0.9 px on average, 3 px at most (hook pitch ~21 px); a homography is needed, a similarity
transform was up to 9 px off because the camera also tilted. Rack frames gave 54-123 agreeing matches, other
scenes at most 7.

```bash
# set the layout and the frame it was drawn on (only its features are kept, no pixels: data/layouts/<id>.ref.npz)
curl -X PUT -H "Content-Type: application/json" --data @live01.layout.json http://localhost:8000/cameras/live01/layout
curl -X PUT --data-binary @live01.jpeg http://localhost:8000/cameras/live01/reference   # or ?latest=true
curl http://localhost:8000/cameras/live01/reference          # reference + last registration
curl -X DELETE http://localhost:8000/cameras/live01/reference  # crop the pictures as they come again
```

Always set the two together: a reference without its own layout warps the pictures onto the wrong frame.
`BIKESLOT_REGISTRATION=0` turns the step off for every camera.

## Code

| Module | Role |
|---|---|
| `registration.py` | reference features, `register` (match + RANSAC + checks), `warp` onto the reference frame |
| `slotcls/crops.py` | polygon -> perspective-rectified patch (`crop_polygon`, `slot_crops`), contact sheet |
| `slotcls/model.py` | `SlotClassifier` (torchvision backbone + linear head), `train`, `evaluate`, save/load, cached `get_model` |
| `slotcls/dataset.py` | label files -> samples; context patches; augmentation (shift/zoom/rotate inside real surroundings, flip, brightness, blur, noise); `CropDataset` |
| `slotcls/smoothing.py` | probability -> vote (unsure band) on top of `occupancy.SlotTracker` (window + hysteresis) |
| `slotcls/pipeline.py` | `analyze_frame(layout, img, camera_id, trackers)` -> `CameraStatus` (+ probabilities, crops, person boxes) |
| `scripts/slotcls_calibrate.py`, `_label.py`, `_train.py`, `_run.py` | the four steps above |
| `tests/test_slotcls.py` | crops, smoothing, demo layout/labels consistency, train/save/load round trip, pipeline with a stub model |
| `tests/test_registration.py` | bumped camera recovered < 1.5 px (also from a 320 px picture), other scenes rejected, no pixels stored, aligned crops, reset after failures |

Label file format (what the label tool writes, what the trainer reads):

```json
{"camera_id": "demo01", "layout": "demo_picture.layout.json",
 "frames": [{"file": "demo_picture.jpeg", "slots": {"demo01-S01": "free", "demo01-S03": "occupied", "...": "..."}}]}
```

Paths are relative to the label file; `unknown` labels are skipped. Crops are cut from the full-resolution
frames at training time, nothing is duplicated on disk; frames stay where they are (and raw frames of the
live camera are still never written by the service: snapshots for labelling have to be saved deliberately,
e.g. with `replay.py` or a `curl` loop against the camera's `/snapshot`).

## Next steps

* Save a few dozen snapshots of the demo scene with different bikes (or of the live camera once it is
  mounted), label them (one click per changed slot), retrain with `--cv frames`.
* When there are a few thousand crops: `--finetune` (whole backbone) and compare.
* The admin UI can draw the polygons directly (`PUT /cameras/{id}/layout` with `slots` and
  `params: {"engine": "classifier"}`); the calibration script is only the quick path for straight racks.
