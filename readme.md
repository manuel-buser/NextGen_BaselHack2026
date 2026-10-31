# Find my Bike Slot

**Team NextGen** – BaselHack 2026
**Challenge:** Find my Bike Slot (Endress+Hauser)
**Tags:** #image recognition, #embedded

## Challenge

How might we help employees quickly find an available bicycle parking slot at our company site?
Bicycle parking areas can become crowded, and employees do not know where free slots are available before arriving.
We explore a digital, physical, or combined solution that makes bicycle parking more convenient, efficient, and transparent.

### Nice-to-haves
- Real-time or estimated slot availability
- Directions to available parking
- A simple mobile or web interface
- Privacy friendly
- A prototype (with sensors even?)

### No-Gos
- Storing faces, license plates or other personal data
- Identifying or tracking individual persons

### Resources
- 10x ESP32 cameras with board
- Reference images of the parking slots
- 100 CHF for AI tools (LLM providers etc.) per Endress team

## Team

| Name |
|------|
| Dennis Xhabrahimi |
| Jevin Jiji Thaiparampil |
| Jiri Käser |
| Manuel Buser |

## Repository structure

- `code/` – all code (firmware, image recognition, web/mobile interface)
- `documentation/` – diagrams, architecture, code-related docs
- `assets/` – videos, presentations, images

## Solution

An ESP32-CAM sends a JPEG snapshot of the parking row every few seconds to a small Python service on a
laptop. The service detects bicycles with YOLO11 (segmentation), maps each bike's ground-contact point to a
slot polygon, smooths the result over a few frames and publishes per-slot occupancy as JSON for the
admin/user web app. A new camera is calibrated once: the service proposes the slot layout (bikes it has
seen, a GPT-vision scene description, painted lines) and an admin confirms or corrects it in the web app.
Persons and licence plates are blurred before anything is shown or uploaded; raw frames never touch the disk.

- Vision pipeline, API contract, ESP32 contract and privacy notes: [documentation/vision-pipeline.md](documentation/vision-pipeline.md)
- Code: [code/bikeslot/](code/bikeslot/) (service), [code/scripts/](code/scripts/) (replay = fake camera, golden evaluation), [code/tests/](code/tests/)
- Golden data set with hand-labelled slots: [assets/pictures/GoldenDataSet/](assets/pictures/GoldenDataSet/)

### Run the vision service

```bash
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn --app-dir code bikeslot.api:app --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs  (API)   http://127.0.0.1:8000/  (debug view)
.venv/Scripts/python code/scripts/replay.py --camera cam01 --source assets/pictures/GoldenDataSet --interval 3 --loop
```

## License

MIT – see [license.txt](license.txt).
