"""Train the free/occupied slot classifier on labelled frames.

    .venv/Scripts/python code/scripts/slotcls_train.py --labels assets/pictures/DemoPictures/demo_picture.labels.json \
        --out assets/models/slotcls_demo.pth --cv slots

Label files are what ``slotcls_label.py`` writes (one label per slot per frame, layout referenced by path).
Several label files (several cameras / days) can be given.  With few frames use ``--cv slots`` (leave-one-
slot-out: how well does the model generalise to a crop it has not seen) or ``--cv frames`` (leave-one-
frame-out) to get an honest number before the final fit on everything.  ``--finetune`` unfreezes the
backbone (for a few thousand crops); the default trains only the linear head on frozen ImageNet features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikeslot import config  # noqa: E402
from bikeslot.slotcls import dataset as ds  # noqa: E402
from bikeslot.slotcls.model import SlotClassifier, evaluate, train  # noqa: E402


def fit(samples, args, val_samples=None, log=print) -> tuple[SlotClassifier, dict]:
    model = SlotClassifier(args.backbone, args.input, pretrained=True)
    tr = ds.CropDataset(samples, args.input, args.margin, args.context, train=True, seed=args.seed)
    va = ds.CropDataset(val_samples, args.input, args.margin, args.context, train=False) if val_samples else None
    hist = train(model, tr, epochs=args.epochs, lr=args.lr, finetune=args.finetune, batch_size=args.batch,
                 device=args.device, val_dataset=va, log=log, seed=args.seed)
    return model, hist


def cross_validate(samples, args) -> None:
    if args.cv == "slots":
        folds = [[i] for i in range(len(samples))]
        name = "slot"
    else:
        frames = sorted({s.frame for s in samples})
        folds = [[i for i, s in enumerate(samples) if s.frame == f] for f in frames]
        name = "frame"
    if len(folds) < 2:
        print("not enough data for cross-validation")
        return
    probs = np.zeros(len(samples))
    for k, held in enumerate(folds):
        held_set = set(held)
        train_s = [s for i, s in enumerate(samples) if i not in held_set]
        val_s = [samples[i] for i in held]
        model, _ = fit(train_s, args, log=lambda *_: None)
        ev = evaluate(model, ds.CropDataset(val_s, args.input, args.margin, args.context))
        for i, p in zip(held, ev["probs"]):
            probs[i] = p
        print(f"  fold {k + 1}/{len(folds)} ({name} held out): {ev['tp'] + ev['tn']}/{ev['n']} correct")
    labels = np.array([s.label for s in samples])
    pred = (probs >= 0.5).astype(int)
    acc = float((pred == labels).mean())
    print(f"\nleave-one-{name}-out: {int((pred == labels).sum())}/{len(samples)} correct ({acc:.1%})")
    wrong = [(s.frame.name, s.slot_id, s.label, round(float(p), 2)) for s, p, ok in zip(samples, probs, pred == labels) if not ok]
    for frame, sid, label, p in wrong:
        print(f"  wrong: {frame} {sid} truth={'occupied' if label else 'free'} p_occ={p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="label JSON files (see slotcls_label.py)")
    ap.add_argument("--val-labels", nargs="*", default=[], help="label files used only for validation")
    ap.add_argument("--out", default=str(config.SLOTCLS_MODEL))
    ap.add_argument("--backbone", default=config.SLOTCLS_BACKBONE)
    ap.add_argument("--input", default=str(config.SLOTCLS_INPUT), help="crop size, e.g. 96 or 64x160")
    ap.add_argument("--margin", type=float, default=config.SLOTCLS_MARGIN)
    ap.add_argument("--context", type=float, default=0.35, help="extra polygon growth kept for shift/zoom augmentation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--finetune", action="store_true", help="also train the backbone (needs many crops)")
    ap.add_argument("--cv", choices=["none", "slots", "frames"], default="none", help="cross-validation report before the final fit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = ds.materialize(ds.load_samples(args.labels), args.input, args.margin, args.context)
    n_occ = sum(s.label for s in samples)
    print(f"{len(samples)} labelled crops from {len({s.frame for s in samples})} frame(s): {n_occ} occupied, {len(samples) - n_occ} free")
    if not samples:
        sys.exit("nothing to train on")
    val_samples = ds.materialize(ds.load_samples(args.val_labels), args.input, args.margin, args.context) if args.val_labels else None

    if args.cv != "none":
        print(f"cross-validation ({args.cv}) ...")
        cross_validate(samples, args)
        print()

    print("final fit on all labelled crops ...")
    model, hist = fit(samples, args, val_samples)
    ev = evaluate(model, ds.CropDataset(samples, args.input, args.margin, args.context))
    print(f"training set: {ev['tp'] + ev['tn']}/{ev['n']} correct  ({hist['seconds']} s)")
    if val_samples:
        ev = evaluate(model, ds.CropDataset(val_samples, args.input, args.margin, args.context))
        print(f"validation set: {ev['tp'] + ev['tn']}/{ev['n']} correct  (fp {ev['fp']}, fn {ev['fn']})")
    model.save(args.out)
    print(f"model saved to {args.out}")


if __name__ == "__main__":
    main()
