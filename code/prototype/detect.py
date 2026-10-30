"""Prototype: run pretrained detectors on the example photos.

Two models:
  - coco:  YOLO11 trained on COCO, finds "bicycle" and "motorcycle" (live detection, pipeline step 3)
  - world: YOLO-World open-vocabulary, finds parking structures from text prompts (setup, steps 1+2)

People are blurred in the saved images. Outputs go to data/out/ (gitignored).

Usage:
  .venv/Scripts/python code/prototype/detect.py [--model coco|world|both]
"""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOWorld

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "assets" / "pictures"
OUT = ROOT / "data" / "out"
MAX_SIDE = 1600  # phone photos are 4000x3000; ESP32-CAM will be much smaller anyway

WORLD_PROMPTS = ["bicycle", "motorcycle", "bicycle rack", "bicycle shelter", "empty parking space"]
COLORS = {"bicycle": (0, 200, 0), "motorcycle": (0, 140, 255), "person": (0, 0, 255)}


def load(path):
    img = cv2.imread(str(path))
    scale = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def blur_people(img, boxes):
    for x1, y1, x2, y2 in boxes:
        roi = img[y1:y2, x1:x2]
        if roi.size:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)


def draw(img, dets):
    for name, conf, (x1, y1, x2, y2) in dets:
        color = COLORS.get(name, (255, 0, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def run(model, tag, conf, keep):
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(IMAGES.glob("*.jpg")):
        img = load(path)
        res = model.predict(img, conf=conf, imgsz=1280, verbose=False)[0]
        dets, people = [], []
        for box in res.boxes:
            name = res.names[int(box.cls)]
            xyxy = tuple(int(v) for v in box.xyxy[0])
            if name == "person":
                people.append(xyxy)
            elif name in keep:
                dets.append((name, float(box.conf), xyxy))
        blur_people(img, people)
        draw(img, dets)
        cv2.imwrite(str(out_dir / path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        counts = {k: sum(d[0] == k for d in dets) for k in keep}
        rows.append({"image": path.name, **counts})
        print(tag, path.name, counts)
    with open(out_dir / "counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["coco", "world", "both"], default="both")
    args = ap.parse_args()

    if args.model in ("coco", "both"):
        run(YOLO("yolo11x.pt"), "coco", conf=0.25, keep=["bicycle", "motorcycle"])
    if args.model in ("world", "both"):
        world = YOLOWorld("yolov8x-worldv2.pt")
        world.set_classes(WORLD_PROMPTS + ["person"])
        run(world, "world", conf=0.1, keep=WORLD_PROMPTS)


if __name__ == "__main__":
    main()
