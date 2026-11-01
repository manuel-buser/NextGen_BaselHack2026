"""Binary free/occupied classifier for slot crops: ImageNet torchvision backbone (MobileNetV3-small) + linear head.

``finetune=False`` trains the head only (a few dozen crops), ``True`` everything. The ``.pth`` holds backbone
name, input size and state dict (head only when frozen: a few KB, the backbone weights come from torchvision).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .. import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BACKBONES = {"mobilenet_v3_small": 576, "mobilenet_v3_large": 960, "resnet18": 512, "efficientnet_b0": 1280}


def parse_size(value) -> tuple[int, int]:
    """96 -> (96, 96); '64x160' -> (64, 160) (width x height)."""
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    s = str(value).lower().replace("*", "x")
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    return int(s), int(s)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    import torchvision.models as tvm

    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}, choose from {sorted(BACKBONES)}")
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = m.features
    elif name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = m.features
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = m.features
    else:  # resnet18
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
    return feats, BACKBONES[name]


class SlotClassifier(nn.Module):
    def __init__(self, backbone: str = config.SLOTCLS_BACKBONE, input_size=config.SLOTCLS_INPUT, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.input_size = parse_size(input_size)
        self.features, dim = _build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 1))
        self.finetuned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.features(x)).flatten(1)
        return self.head(f).squeeze(1)       # logit of "occupied"

    # ------------------------------------------------------------------ preprocessing
    def to_tensor(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        """BGR uint8 crops (any size) -> normalised float tensor (N, 3, H, W) at the model's input size."""
        import cv2

        w, h = self.input_size
        out = np.empty((len(crops), h, w, 3), dtype=np.float32)
        for i, c in enumerate(crops):
            if c.shape[1] != w or c.shape[0] != h:
                c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
            out[i] = c[:, :, ::-1].astype(np.float32) / 255.0
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(out.transpose(0, 3, 1, 2)))

    @torch.no_grad()
    def predict_proba(self, crops: Sequence[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """P(occupied) for every crop."""
        if len(crops) == 0:
            return np.zeros(0, dtype=np.float32)
        self.eval()
        device = next(self.parameters()).device
        probs = []
        for i in range(0, len(crops), batch_size):
            x = self.to_tensor(crops[i:i + batch_size]).to(device)
            probs.append(torch.sigmoid(self(x)).float().cpu().numpy())
        return np.concatenate(probs)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict() if self.finetuned else {k: v for k, v in self.state_dict().items() if k.startswith("head.")}
        torch.save({"backbone": self.backbone_name, "input_size": list(self.input_size), "finetuned": self.finetuned,
                    "state_dict": state, "saved_at": time.time()}, path)

    @classmethod
    def load(cls, path: Path | str, device: Optional[str] = None) -> "SlotClassifier":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(ckpt["backbone"], tuple(ckpt["input_size"]), pretrained=not ckpt.get("finetuned", False))
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected or any(not k.startswith("features.") for k in missing):
            raise RuntimeError(f"checkpoint does not fit {ckpt['backbone']}: missing {missing[:3]}, unexpected {unexpected[:3]}")
        model.finetuned = bool(ckpt.get("finetuned", False))
        model.to(pick_device(device))
        model.eval()
        return model


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------- training

def train(model: SlotClassifier, dataset, epochs: int = 40, lr: Optional[float] = None, finetune: bool = False,
          batch_size: int = 64, device: Optional[str] = None, val_dataset=None, log=print, seed: int = 0,
          min_per_epoch: int = 256) -> dict:
    """Fit the head (or everything with ``finetune``) with BCE loss and class balancing.

    ``dataset`` yields (tensor 3xHxW, label float); see ``dataset.CropDataset``.  Returns a small history dict.
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    device = pick_device(device)
    model.to(device)
    for p in model.features.parameters():
        p.requires_grad = finetune
    model.finetuned = finetune or model.finetuned
    params = [p for p in model.parameters() if p.requires_grad]
    lr = lr or (1e-4 if finetune else 2e-3)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    labels = np.array([float(y) for _, y in dataset.items()]) if hasattr(dataset, "items") else None
    pos_weight = None
    if labels is not None and labels.sum() > 0 and (1 - labels).sum() > 0:
        pos_weight = torch.tensor([(1 - labels).sum() / labels.sum()], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # tiny label sets: draw at least ``min_per_epoch`` (augmented) crops per epoch so the head gets enough steps
    n_draw = max(len(dataset), min_per_epoch)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=n_draw)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    history: dict = {"loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if not finetune:
            model.features.eval()          # keep BatchNorm statistics of the pretrained backbone
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        history["loss"].append(total / max(1, n))
        msg = f"epoch {ep + 1:3d}/{epochs}  loss {total / max(1, n):.4f}"
        if val_dataset is not None and (ep % 5 == 4 or ep == epochs - 1):
            acc = evaluate(model, val_dataset, device)["accuracy"]
            history["val_acc"].append(acc)
            msg += f"  val acc {acc:.3f}"
        log(msg)
    history["seconds"] = round(time.time() - t0, 1)
    model.eval()
    return history


@torch.no_grad()
def evaluate(model: SlotClassifier, dataset, device: Optional[str] = None, threshold: float = 0.5) -> dict:
    from torch.utils.data import DataLoader

    device = pick_device(device)
    model.to(device).eval()
    probs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0):
        probs.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(probs), np.concatenate(ys).astype(int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": int(len(y)), "accuracy": (tp + tn) / max(1, len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "probs": p, "labels": y}


_loaded: dict = {}


def get_model(path: Path | str = config.SLOTCLS_MODEL) -> SlotClassifier:
    """Process-wide cached classifier loaded from ``path`` (reloaded when the file changes)."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime if path.exists() else None)
    if _loaded.get("key") != key:
        if not path.exists():
            raise FileNotFoundError(f"no slot classifier at {path} - train one with code/scripts/slotcls_train.py")
        _loaded["model"] = SlotClassifier.load(path)
        _loaded["key"] = key
    return _loaded["model"]


def info(path: Path | str = config.SLOTCLS_MODEL) -> dict:
    path = Path(path)
    m = _loaded.get("model")
    return {"model_path": str(path), "available": path.exists(), "loaded": m is not None,
            "backbone": m.backbone_name if m else None, "input_size": list(m.input_size) if m else None,
            "finetuned": m.finetuned if m else None,
            "device": str(next(m.parameters()).device) if m else None}
