"""
train_cedar_yolo.py

CEDAR-YOLO in the paper is called SigYOLO / AsymYOLO (sig_yolo_* runs) in the code.

Training script for SigYOLO (AsymYOLO) —
asymmetric single-scale YOLO for CBRS interference detection on spectrogram images.

No neck, factored asymmetric detection head, independent freq/time DFL.
Variants: v8 (C2f), v11 (C3k2 + PSA), v26 (A2C2f SE-attention).

Usage:
    python scripts/data/split_yolo_dataset.py          # build yolo_split/ first
    python scripts/cedar/train_cedar_yolo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import sys
# VERSION WARNING:
# The trained model depends on the exact yolo_split PNG rendering distribution.
# Dataset PNG generation, training, SigMF raw conversion, and inference
# must use the same rendering stack:
#   matplotlib==3.10.8
#   numpy==2.2.4
#   pillow==11.2.1
# This version set has passed virtual-PNG pixel validation against yolo_split.

import math
import random
import hashlib
import numpy as np
import warnings
from pathlib import Path

# LambdaLR calls step() once in __init__ before any optimizer.step() —
# suppress the false-positive "step before optimizer.step" warning it emits.
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
)
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml as _yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.data.split_yolo_dataset import validate_split_yaml
from cbrs.cedar_yolo import (
    AsymDetectLoss,
    asym_non_max_suppression,
    build_asym_yolo,
)
from cbrs.augment import ModelEMA, SigYOLOAugment
from cbrs.training_state import completed, advance, device_name

# ─────────────────────────────────────────────────────────────────────────────
# Config  (edit these globals before running)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT         = Path(__file__).resolve().parents[2]
OUTPUT_DIR         = PROJECT_ROOT / "yolo_split"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models_retrained"

RUN_MODE   = "sweep"   # "train" or "sweep"
EPOCHS     = 100
BATCH_SIZE = 8
IMG_SIZE   = 640
DEVICE     = None      # None = auto, "0" = GPU 0, "cpu" = CPU
WORKERS    = 0         # 0 = main-process loading (faster on Windows; spawn overhead kills perf)
PATIENCE   = 20        # early-stop if no mAP50 improvement for N epochs
SAVE_PERIOD = 10       # save a numbered checkpoint every N epochs
CACHE_IMAGES = True    # pre-load entire train+val set into RAM before first epoch

# Single-model config  (RUN_MODE = "train")
VARIANT  = "v8"
SIZE     = "n"
RUN_NAME = None        # None → auto-generated, e.g. "sig_yolo_v8n"

# Sweep config  (RUN_MODE = "sweep")
# Default: the three paper models sig_yolo_v8{n,s,x}_f20t20_c100.
# Full paper sweep (108 runs): SWEEP_FM_FREQS = SWEEP_FM_TIMES = [20, 40, 80],
# SWEEP_COVERAGE_LEVELS = [0.25, 0.50, 1.00, 1.50].
SWEEP_VARIANTS        = ["v8"]
SWEEP_SCALES          = ["n", "s", "x"]
# fm sweep: input_w / fm must be 2^k (input=640 → valid: 20, 40, 80, 160, …)
SWEEP_FM_FREQS        = [20]
SWEEP_FM_TIMES        = [20]
# reg_max = round(coverage * fm); coverage = fraction of image spanned per side
SWEEP_COVERAGE_LEVELS = [1.00]
SWEEP_PARALLEL        = 1
SKIP_MISSING_LOCAL    = False
SKIP_COMPLETED        = True   # explicit completed/early-stop state only
SEED = 0
MIN_DELTA = 1e-6

# Model / head parameters  (used by RUN_MODE = "train" only)
FM_FREQ      = 80    # feature-map width  (freq columns);  input_w / fm_freq must be 2^k
FM_TIME      = 20    # feature-map height (time rows);     input_h / fm_time must be 2^k
REG_MAX_FREQ = 28    # DFL bins for left/right offsets
REG_MAX_TIME = 10    # DFL bins for top/bottom offsets

# Hyper-parameters  (ultralytics v8 defaults)
LR0           = 0.01    # initial learning rate  (Ultralytics SGD default)
LRF           = 0.01    # final LR ratio  (LR_final = LR0 * LRF)
MOMENTUM      = 0.937   # SGD momentum  (Ultralytics default, post-warmup)
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 3
BOX_GAIN      = 7.5
CLS_GAIN      = 0.5
DFL_GAIN      = 1.5

# Augmentation config (matches ultralytics defaults; mosaic / auto_augment excluded)
USE_AUGMENT   = False  # global switch; False = no augmentation (consistent with test_yolo.py)
AUG_FLIPLR    = 0.5
AUG_SCALE     = 0.5
AUG_TRANSLATE = 0.1
AUG_HSV_H     = 0.015
AUG_HSV_S     = 0.7
AUG_HSV_V     = 0.4
AUG_ERASING   = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Data-yaml parser  (handles ultralytics yaml layout)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


def _parse_data_yaml(
    yaml_path: Path,
) -> Tuple[Path, Path, int, List[str]]:
    """
    Parse ultralytics data.yaml → (train_img_dir, val_img_dir, nc, class_names).
    Handles optional 'path' key for relative resolution.
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    # Resolve dataset root:
    #   absolute path  → use as-is
    #   relative (incl. ".") → relative to yaml file's directory (portable)
    #   missing        → yaml file's directory
    if "path" in data:
        _p = Path(data["path"]).expanduser()
        base = _p.resolve() if _p.is_absolute() else (yaml_path.parent / _p).resolve()
    else:
        base = yaml_path.parent.resolve()
    train_img = _resolve(base, data["train"])
    val_img   = _resolve(base, data.get("val", data["train"]))

    # names can be a list ["LTE","Radar"] or a dict {0:"LTE", 1:"Radar"}
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        n = max(raw_names.keys()) + 1
        names = [raw_names.get(i, str(i)) for i in range(n)]
    else:
        names = list(raw_names)
    nc = int(data["nc"]) if "nc" in data else len(names)

    return train_img, val_img, nc, names


def _label_dir(img_dir: Path) -> Path:
    """
    Resolve label directory for an image directory.
    Supports two common YOLO layouts:
      split/images/  →  split/labels/   (e.g. train/images → train/labels)
      images/split/  →  labels/split/   (e.g. images/train → labels/train)
    The sibling layout is tried first.
    """
    sibling = img_dir.parent / "labels"
    if sibling.exists():
        return sibling
    return img_dir.parent.parent / "labels" / img_dir.name


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SigYOLODataset(Dataset):
    """
    YOLO-format image/label dataset for SigYOLO training.

    Labels: one .txt per image — rows of ``class_id cx cy w h`` (normalised 0–1).
    Returns (img [3, H, W]  float32 in [0, 1],
             labels [N, 5]  float32: cls x1 y1 x2 y2 in pixels).
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        img_dir:   Path,
        label_dir: Path,
        img_size:  int  = 640,
        augment:   "bool | SigYOLOAugment | None" = False,
        cache:     bool = False,
    ):
        self.img_size = img_size
        self.lbl_dir  = label_dir
        self.imgs     = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in self._EXTS
        )
        if not self.imgs:
            raise RuntimeError(f"No images found in {img_dir}")
        if isinstance(augment, SigYOLOAugment):
            self._augmenter: Optional[SigYOLOAugment] = augment
        elif augment:
            self._augmenter = SigYOLOAugment()
        else:
            self._augmenter = None

        self._cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if cache:
            self._build_cache(img_size, label_dir)

    def _build_cache(self, img_size: int, label_dir: Path) -> None:
        from PIL import Image as _PIL
        import torchvision.transforms.functional as TF

        n = len(self.imgs)
        mb = n * img_size * img_size * 3 * 4 / 1024 / 1024
        print(f"  Caching {n} images (~{mb:.0f} MB) ...", flush=True)
        cache = []
        for img_path in tqdm(self.imgs, desc="  cache", unit="img", file=sys.stdout, leave=False):
            img = _PIL.open(img_path).convert("RGB")
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), _PIL.BILINEAR)
            img_t = TF.to_tensor(img)

            lbl_path = label_dir / (img_path.stem + ".txt")
            rows: List[List[float]] = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls        = float(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = max(0.0, (cx - w / 2) * img_size)
                        y1 = max(0.0, (cy - h / 2) * img_size)
                        x2 = min(float(img_size), (cx + w / 2) * img_size)
                        y2 = min(float(img_size), (cy + h / 2) * img_size)
                        if x2 > x1 and y2 > y1:
                            rows.append([cls, x1, y1, x2, y2])
            lbl_t = (
                torch.tensor(rows, dtype=torch.float32)
                if rows
                else torch.zeros((0, 5), dtype=torch.float32)
            )
            cache.append((img_t, lbl_t))
        self._cache = cache
        print(f"  Cache ready ({n} images).", flush=True)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            img_t, lbl_t = self._cache[idx]
            img_t = img_t.clone()
            lbl_t = lbl_t.clone()
        else:
            from PIL import Image as _PIL
            import torchvision.transforms.functional as TF

            img_path = self.imgs[idx]
            img = _PIL.open(img_path).convert("RGB")
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size), _PIL.BILINEAR)
            img_t = TF.to_tensor(img)

            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            rows: List[List[float]] = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls        = float(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = max(0.0, (cx - w / 2) * self.img_size)
                        y1 = max(0.0, (cy - h / 2) * self.img_size)
                        x2 = min(float(self.img_size), (cx + w / 2) * self.img_size)
                        y2 = min(float(self.img_size), (cy + h / 2) * self.img_size)
                        if x2 > x1 and y2 > y1:
                            rows.append([cls, x1, y1, x2, y2])
            lbl_t = (
                torch.tensor(rows, dtype=torch.float32)
                if rows
                else torch.zeros((0, 5), dtype=torch.float32)
            )

        if self._augmenter is not None:
            img_t, lbl_t = self._augmenter(img_t, lbl_t)

        return img_t, lbl_t


def spec_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate images and variable-length label tensors.
    Prepends batch index → targets [N, 6]: (batch_idx cls x1 y1 x2 y2).
    Matches AsymDetectLoss target format exactly.
    """
    imgs, label_list = zip(*batch)
    imgs_t = torch.stack(imgs)
    parts  = []
    for i, lbl in enumerate(label_list):
        if lbl.shape[0]:
            bidx = torch.full((lbl.shape[0], 1), float(i))
            parts.append(torch.cat([bidx, lbl], dim=1))
    targets = torch.cat(parts, dim=0) if parts else torch.zeros((0, 6))
    return imgs_t, targets


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser  (mirrors ultralytics/engine/trainer.py :: build_optimizer)
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, momentum: float,
) -> optim.Optimizer:
    """
    SGD with Nesterov momentum — matches Ultralytics build_optimizer default
    (the `optimizer='auto'` SGD branch, used for COCO-scale training).
    Three parameter groups:
      g0  BN weights    (no weight decay)
      g1  Conv weights  (with weight decay)
      g2  all biases    (no weight decay)
    """
    g0, g1, g2 = [], [], []
    for m in model.modules():
        if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
            g2.append(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            g0.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            g1.append(m.weight)
    opt = optim.SGD(g2, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({"params": g0, "weight_decay": 0.0})
    opt.add_param_group({"params": g1, "weight_decay": weight_decay})
    return opt


def _build_scheduler(
    opt: optim.Optimizer, epochs: int, warmup_epochs: int, lrf: float,
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup + cosine decay.
    Mirrors ultralytics/utils/torch_utils.py :: one_cycle.
    """
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return lrf + 0.5 * (1.0 - lrf) * (1.0 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device, scaler, ema=None, pbar=None, batch_callback=None):
    model.train()
    total = box = cls = dfl = n = 0.0
    n_batches = len(loader)
    for batch_idx, (imgs, targets) in enumerate(loader):
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, parts = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, parts = criterion(model(imgs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total += loss.item()
        box   += parts["box"].item()
        cls   += parts["cls"].item()
        dfl   += parts["dfl"].item()
        n     += 1.0

        if batch_callback is not None:
            batch_callback(batch_idx + 1, n_batches)

    n = max(n, 1.0)
    return {"total": total / n, "box": box / n, "cls": cls / n, "dfl": dfl / n}


# ─────────────────────────────────────────────────────────────────────────────
# Validation / mAP@50
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """[N, 4] vs [M, 4] xyxy  →  [N, M] IoU."""
    inter = (
        b1[:, None, 2:].minimum(b2[None, :, 2:])
        - b1[:, None, :2].maximum(b2[None, :, :2])
    ).clamp(0).prod(-1)
    a1 = (b1[:, 2:] - b1[:, :2]).prod(-1)
    a2 = (b2[:, 2:] - b2[:, :2]).prod(-1)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


def _ap_from_pr(recall: List[float], precision: List[float]) -> float:
    """Area under PR curve — 11-point interpolation (PASCAL VOC)."""
    mrec = [0.0] + recall + [1.0]
    mpre = [0.0] + precision + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum(
        max((p for r, p in zip(mrec, mpre) if r >= t), default=0.0) / 11.0
        for t in (i * 0.1 for i in range(11))
    )


@torch.no_grad()
def validate(
    model, loader, nc: int, device,
    conf: float = 0.001, iou_nms: float = 0.6, iou_match: float = 0.5,
) -> float:
    """Compute mAP@50. Returns scalar in [0, 1]."""
    model.eval()

    all_preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    all_gts:   List[Tuple[torch.Tensor, torch.Tensor]] = []

    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        B       = imgs.shape[0]

        y, _ = model(imgs)
        dets  = asym_non_max_suppression(y, conf_thres=conf, iou_thres=iou_nms, max_det=300)

        for i in range(B):
            det = dets[i].cpu()
            if det.shape[0]:
                all_preds.append((det[:, :4], det[:, 4], det[:, 5].long()))
            else:
                all_preds.append((
                    torch.zeros(0, 4),
                    torch.zeros(0),
                    torch.zeros(0, dtype=torch.long),
                ))
            mask = targets[:, 0].long() == i
            gt   = targets[mask].cpu()
            if gt.shape[0]:
                all_gts.append((gt[:, 2:6], gt[:, 1].long()))
            else:
                all_gts.append((torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)))

    aps: List[float] = []
    for c in range(nc):
        tp_fp: List[Tuple[int, int]] = []
        n_gt  = 0

        for (pb, ps, pc), (gb, gc) in zip(all_preds, all_gts):
            gb_c = gb[gc == c]
            n_gt += gb_c.shape[0]
            mask_c = pc == c
            pb_c   = pb[mask_c]
            ps_c   = ps[mask_c]
            if pb_c.shape[0] == 0:
                continue
            order   = ps_c.argsort(descending=True)
            pb_c    = pb_c[order]
            matched = torch.zeros(gb_c.shape[0], dtype=torch.bool)
            for bi in range(pb_c.shape[0]):
                if gb_c.shape[0] == 0:
                    tp_fp.append((0, 1))
                    continue
                iou = _pairwise_iou(pb_c[bi : bi + 1], gb_c)[0]
                best_v, best_j = iou.max(0)
                if best_v.item() >= iou_match and not matched[best_j]:
                    matched[best_j] = True
                    tp_fp.append((1, 0))
                else:
                    tp_fp.append((0, 1))

        if n_gt == 0 or not tp_fp:
            aps.append(0.0)
            continue
        tp   = torch.tensor([t for t, _ in tp_fp], dtype=torch.float32).cumsum(0)
        fp   = torch.tensor([f for _, f in tp_fp], dtype=torch.float32).cumsum(0)
        rec  = (tp / (n_gt + 1e-7)).tolist()
        prec = (tp / (tp + fp + 1e-7)).tolist()
        aps.append(_ap_from_pr(rec, prec))

    return sum(aps) / max(len(aps), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Run-name helper
# ─────────────────────────────────────────────────────────────────────────────

def run_name_for(
    variant: str,
    size: str,
    fm_freq: Optional[int] = None,
    fm_time: Optional[int] = None,
    coverage: Optional[float] = None,
) -> str:
    """Build a unique run name.
    train mode:  ("v8","n")              → "sig_yolo_v8n"
    sweep mode:  ("v8","n",80,20,0.50)  → "sig_yolo_v8n_f80t20_c50"
    """
    base = f"sig_yolo_{variant.lower()}{size.lower()}"
    if fm_freq is not None and fm_time is not None and coverage is not None:
        return f"{base}_f{fm_freq}t{fm_time}_c{round(coverage * 100):d}"
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Run-completion check
# ─────────────────────────────────────────────────────────────────────────────

def completed_epochs(run_name: str) -> int:
    """Return how many epochs the run completed (0 if not started or unreadable).

    Reads last.pt and returns ckpt['epoch'] + 1  (epoch is 0-indexed in ckpt).
    """
    last_pt = TRAINED_MODELS_DIR / run_name / "weights" / "last.pt"
    if not last_pt.exists():
        return 0
    try:
        ck = torch.load(str(last_pt), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", -1)) + 1  # 0-indexed → count
    except Exception:
        return 0


def is_run_completed(run_name: str, target_epochs: int = EPOCHS) -> bool:
    path = TRAINED_MODELS_DIR / run_name / "weights" / "last.pt"
    if not path.exists():
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return completed(ck, target_epochs)


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    yaml_path=None,
    variant=None,
    size=None,
    run_name=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    patience=None,
    save_period=None,
    fm_freq=None,
    fm_time=None,
    stride_freq=None,
    stride_time=None,
    reg_max_freq=None,
    reg_max_time=None,
    cached_train_ds=None,
    cached_val_ds=None,
    progress_queue=None,
    progress_job_id=None,
    quiet=False,
):
    """Train one SigYOLO model.  Interface mirrors train_yolo.py :: train_model."""
    yaml_path   = yaml_path   or validate_split_yaml(output_dir=OUTPUT_DIR)
    variant     = variant     or VARIANT
    size        = size        or SIZE
    run_name    = run_name    or run_name_for(variant, size)
    epochs      = epochs      or EPOCHS
    batch_size  = batch_size  or BATCH_SIZE
    img_size    = img_size    or IMG_SIZE
    device      = DEVICE      if device is None else device
    patience     = PATIENCE     if patience is None else patience
    save_period  = SAVE_PERIOD  if save_period is None else save_period
    fm_freq      = FM_FREQ      if fm_freq is None else fm_freq
    fm_time      = FM_TIME      if fm_time is None else fm_time
    reg_max_freq = REG_MAX_FREQ if reg_max_freq is None else reg_max_freq
    reg_max_time = REG_MAX_TIME if reg_max_time is None else reg_max_time
    # stride_freq / stride_time left as None → build_asym_yolo defaults to input/fm

    if not yaml_path.exists():
        print(f"Error: data.yaml not found: {yaml_path}")
        print("Please run first: python scripts/data/split_yolo_dataset.py")
        sys.exit(1)

    dev = torch.device(
        device_name(device) if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seed = SEED + int(hashlib.sha256(run_name.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_img, val_img, nc, names = _parse_data_yaml(yaml_path)
    train_lbl = _label_dir(train_img)
    val_lbl   = _label_dir(val_img)

    save_dir = TRAINED_MODELS_DIR / run_name
    wt_dir   = save_dir / "weights"
    wt_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 60)
        print("SigYOLO training")
        print("=" * 60)
        print(f"  Model   : {run_name}  (AsymYOLO {variant.upper()}-{size.upper()})")
        print(f"  Classes : {nc}  {names}")
        print(f"  Epochs  : {epochs}  |  Batch: {batch_size}  |  Size: {img_size}")
        print(f"  Device  : {dev}")
        print(f"  Patience: {patience}  |  Save period: {save_period}")
        print(f"  Data    : {yaml_path}")
        print(f"  Output  : {save_dir}")

    aug = SigYOLOAugment(
        fliplr=AUG_FLIPLR, scale=AUG_SCALE, translate=AUG_TRANSLATE,
        hsv_h=AUG_HSV_H, hsv_s=AUG_HSV_S, hsv_v=AUG_HSV_V, erasing=AUG_ERASING,
    ) if USE_AUGMENT else None

    nw = WORKERS
    if cached_train_ds is not None:
        train_ds = cached_train_ds
        train_ds._augmenter = aug
    else:
        train_ds = SigYOLODataset(train_img, train_lbl, img_size=img_size, augment=aug,
                                  cache=CACHE_IMAGES)
    if cached_val_ds is not None:
        val_ds = cached_val_ds
    else:
        val_ds = SigYOLODataset(val_img, val_lbl, img_size=img_size, augment=False,
                                cache=CACHE_IMAGES)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=spec_collate_fn, num_workers=nw,
        pin_memory=(dev.type == "cuda"), drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=spec_collate_fn, num_workers=nw,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(nw > 0),
    )

    build_kw = dict(
        fm_freq=fm_freq, fm_time=fm_time,
        reg_max_freq=reg_max_freq, reg_max_time=reg_max_time,
        input_w=img_size, input_h=img_size,
    )
    if stride_freq is not None:
        build_kw["stride_freq"] = stride_freq
    if stride_time is not None:
        build_kw["stride_time"] = stride_time
    model = build_asym_yolo(variant, size, nc=nc, **build_kw).to(dev)

    ema = ModelEMA(model)

    criterion = AsymDetectLoss(
        model, box_gain=BOX_GAIN, cls_gain=CLS_GAIN, dfl_gain=DFL_GAIN,
    ).to(dev)

    optimizer = _build_optimizer(
        model, lr=LR0, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM,
    )
    scheduler = _build_scheduler(
        optimizer, epochs=epochs, warmup_epochs=WARMUP_EPOCHS, lrf=LRF,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if dev.type == "cuda" and torch.cuda.is_available()
        else None
    )

    if progress_queue is not None:
        progress_queue.put(("start", progress_job_id, {"total": epochs}))

    best_map50 = float('-inf')
    no_improve = 0
    start_epoch = 0
    last_path = wt_dir / 'last.pt'
    if last_path.exists():
        previous = torch.load(last_path, map_location='cpu', weights_only=False)
        required = {'scheduler', 'rng', 'target_epochs', 'no_improve', 'best_map50', 'training_complete'}
        if not required.issubset(previous):
            raise RuntimeError('Legacy checkpoint lacks resume state. Choose a new TRAINED_MODELS_DIR; the old run will not be overwritten.')
        expected = dict(target_epochs=epochs, variant=variant, size=size, nc=nc,
                        fm_freq=fm_freq, fm_time=fm_time, reg_max_freq=reg_max_freq,
                        reg_max_time=reg_max_time, img_size=img_size)
        if any(previous.get(k) != v for k,v in expected.items()):
            raise ValueError('Training configuration differs from last.pt; use a new output directory.')
        if completed(previous, epochs):
            return previous['best_map50']
        model.load_state_dict(previous['model'])
        ema.ema.load_state_dict(previous['ema'])
        ema.updates = previous.get('ema_updates', 0)
        optimizer.load_state_dict(previous['opt'])
        scheduler.load_state_dict(previous['scheduler'])
        if scaler is not None and previous.get('scaler'):
            scaler.load_state_dict(previous['scaler'])
        rng = previous['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng.get('cuda'):
            torch.cuda.set_rng_state_all(rng['cuda'])
        best_map50, no_improve = previous['best_map50'], previous['no_improve']
        start_epoch = previous['epoch'] + 1
        print(f'Resuming {run_name} from epoch {start_epoch + 1}')

    for epoch in range(start_epoch, epochs):
        if progress_queue is not None:
            progress_queue.put(("epoch_start", progress_job_id, {"current": epoch + 1, "total": epochs}))

        def _batch_cb(bi, bt, _ep=epoch + 1, _pq=progress_queue, _jid=progress_job_id):
            if _pq is not None:
                _pq.put(("batch", _jid, {"epoch": _ep, "batch": bi, "total": bt}))

        losses = _train_epoch(
            model, train_loader, optimizer, criterion, dev, scaler, ema,
            batch_callback=_batch_cb if progress_queue is not None else None,
        )
        scheduler.step()
        map50  = validate(ema.ema, val_loader, nc, dev)

        if not quiet:
            print(
                f"[{epoch + 1:3d}/{epochs}]  "
                f"loss={losses['total']:.4f}  "
                f"box={losses['box']:.4f}  cls={losses['cls']:.4f}  dfl={losses['dfl']:.4f}  "
                f"mAP50={map50:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        save_best = map50 >= best_map50
        best_map50, no_improve = advance(map50, best_map50, no_improve, MIN_DELTA)
        early_stop = patience > 0 and no_improve >= patience
        finished = early_stop or epoch + 1 >= epochs
        ckpt = {
            'training_complete': finished,
            'stop_reason': 'early_stopping' if early_stop else ('epoch_budget' if finished else None),
            'target_epochs': epochs,
            'best_map50': best_map50,
            'no_improve': no_improve,
            'seed': seed,
            'min_delta': MIN_DELTA,
            'patience': patience,
            'batch_size': batch_size,
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'ema_updates': ema.updates,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "epoch":        epoch,
            "model":        model.state_dict(),
            "ema":          ema.ema.state_dict(),
            "opt":          optimizer.state_dict(),
            "map50":        map50,
            "variant":      variant,
            "size":         size,
            "nc":           nc,
            "fm_freq":      fm_freq,
            "fm_time":      fm_time,
            "reg_max_freq": reg_max_freq,
            "reg_max_time": reg_max_time,
            "img_size":     img_size,
        }
        temporary = wt_dir / "last.pt.tmp"
        torch.save(ckpt, temporary)
        temporary.replace(wt_dir / "last.pt")

        if save_best:
            torch.save(ckpt, wt_dir / "best.pt")

        if save_period > 0 and (epoch + 1) % save_period == 0:
            torch.save(ckpt, wt_dir / f"epoch{epoch + 1}.pt")

        if progress_queue is not None:
            progress_queue.put(("epoch", progress_job_id, {
                "current": epoch + 1,
                "total":   epochs,
                "loss":    losses["total"],
                "map50":   map50,
            }))

        if early_stop:
            if not quiet:
                print(f"Early stop: no improvement for {patience} epochs.")
            break

    if progress_queue is not None:
        progress_queue.put(("done", progress_job_id, {}))

    if not quiet:
        print(f"\nTraining complete.  Best mAP50={best_map50:.4f}")
        print(f"Best model: {wt_dir / 'best.pt'}")

    return best_map50


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_models(
    yaml_path=None,
    versions=None,
    scales=None,
    fm_freqs=None,
    fm_times=None,
    coverage_levels=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    parallel=SWEEP_PARALLEL,
    skip_missing_local=False,
):
    """Sweep SigYOLO variants × scales × (fm_freq, fm_time) × coverage.

    Each job trains one (variant, size, fm_freq, fm_time, coverage) combination.
    reg_max_freq = round(coverage * fm_freq)
    reg_max_time = round(coverage * fm_time)
    """
    versions        = versions        or SWEEP_VARIANTS
    scales          = scales          or SWEEP_SCALES
    fm_freqs        = fm_freqs        or SWEEP_FM_FREQS
    fm_times        = fm_times        or SWEEP_FM_TIMES
    coverage_levels = coverage_levels or SWEEP_COVERAGE_LEVELS

    if yaml_path is None:
        yaml_path = validate_split_yaml(output_dir=OUTPUT_DIR)

    # planned: (v, s, fm_freq, fm_time, reg_max_freq, reg_max_time, run_name)
    planned = []
    for v in versions:
        for s in scales:
            for ff in fm_freqs:
                for ft in fm_times:
                    for cov in coverage_levels:
                        rmf = round(cov * ff)
                        rmt = round(cov * ft)
                        rn  = run_name_for(v, s, ff, ft, cov)
                        planned.append((v, s, ff, ft, rmf, rmt, rn))

    print("=" * 70)
    print("SigYOLO sweep plan")
    print("=" * 70)
    print(f"Data:       {yaml_path}")
    print(f"Output:     {TRAINED_MODELS_DIR}")
    print(f"Variants:   {versions}  ×  scales {scales}")
    print(f"FM freq:    {fm_freqs}")
    print(f"FM time:    {fm_times}")
    print(f"Coverage:   {[f'{c * 100:.0f}%' for c in coverage_levels]}")
    print(f"Total jobs: {len(planned)}")
    print("=" * 70)

    # Pre-build cached datasets once; all jobs share the same tensors (augmenter is swapped per job)
    _cached_train_ds: Optional[SigYOLODataset] = None
    _cached_val_ds:   Optional[SigYOLODataset] = None
    if CACHE_IMAGES:
        _img_size = img_size or IMG_SIZE
        train_img, val_img, _, _ = _parse_data_yaml(yaml_path)
        train_lbl = _label_dir(train_img)
        val_lbl   = _label_dir(val_img)
        print("Building image cache (shared across all sweep jobs) ...")
        _cached_train_ds = SigYOLODataset(train_img, train_lbl, img_size=_img_size,
                                          augment=False, cache=True)
        _cached_val_ds   = SigYOLODataset(val_img,   val_lbl,   img_size=_img_size,
                                          augment=False, cache=True)

    # job tuple: (index, v, s, fm_freq, fm_time, reg_max_freq, reg_max_time, run_name)
    _RN = 7
    jobs    = []
    skipped = []
    for i, item in enumerate(planned):
        v, s, ff, ft, rmf, rmt, rn = item
        if skip_missing_local and not (TRAINED_MODELS_DIR / rn).exists():
            skipped.append(rn)
            print(f"[{i + 1}/{len(planned)}] Skip (no local weight): {rn}")
            continue
        if SKIP_COMPLETED and is_run_completed(rn, epochs or EPOCHS):
            done_ep = completed_epochs(rn)
            skipped.append(rn)
            print(f"[{i + 1}/{len(planned)}] Skip (done, {done_ep} epochs): {rn}")
            continue
        jobs.append((i + 1, v, s, ff, ft, rmf, rmt, rn))

    failures  = []
    completed = 0
    worker_count = max(1, min(parallel, len(jobs))) if jobs else 0
    print(f"Pending jobs: {len(jobs)}  |  Skipped: {len(skipped)}  |  Workers: {worker_count}")

    if not jobs:
        print("No sweep jobs to run.")
        return failures

    progress_queue: Queue = Queue()

    def drain_progress(pbar, last_epochs, total_epochs_by_job, active_jobs):
        while True:
            try:
                event, job_id, payload = progress_queue.get_nowait()
            except Empty:
                break
            if event == "start":
                total = payload.get("total")
                active_jobs.add(job_id)
                if total:
                    total = int(total)
                    old_total = total_epochs_by_job.get(job_id, epochs or EPOCHS)
                    total_epochs_by_job[job_id] = total
                    if total != old_total:
                        pbar.total += total - old_total
                        pbar.refresh()
                tqdm.write(f"\n[start] {job_id}", file=sys.stdout)
            elif event == "epoch_start":
                current  = payload.get("current", "?")
                ep_total = payload.get("total") or total_epochs_by_job.get(job_id, epochs or EPOCHS)
                tqdm.write(
                    f"  {job_id}  ep {current:3d}/{ep_total}  running...",
                    file=sys.stdout,
                )
            elif event == "batch":
                ep = payload.get("epoch", "?")
                bi = payload.get("batch", 0)
                bt = payload.get("total", 0)
                pbar.set_description(f"ep {ep} batch {bi}/{bt}")
            elif event == "epoch":
                current = payload.get("current")
                if current is None:
                    pbar.update(1)
                    continue
                delta = current - last_epochs.get(job_id, 0)
                if delta > 0:
                    pbar.update(delta)
                    last_epochs[job_id] = current
                ep_total = total_epochs_by_job.get(job_id, epochs or EPOCHS)
                loss  = payload.get("loss")
                map50 = payload.get("map50")
                if loss is not None and map50 is not None:
                    tqdm.write(
                        f"  {job_id}  ep {current:3d}/{ep_total}"
                        f"  loss={loss:.4f}  mAP50={map50:.4f}",
                        file=sys.stdout,
                    )
            elif event == "done":
                total     = total_epochs_by_job.get(job_id, epochs or EPOCHS)
                remaining = total - last_epochs.get(job_id, 0)
                if remaining > 0:
                    pbar.update(remaining)
                    last_epochs[job_id] = total
                active_jobs.discard(job_id)
                tqdm.write(f"[done]  {job_id}", file=sys.stdout)
            elif event == "message":
                tqdm.write(payload.get("text", ""), file=sys.stdout)
            if active_jobs:
                pbar.set_postfix_str("active: " + ", ".join(sorted(active_jobs)))
            else:
                pbar.set_postfix_str("")

    def run_job(job):
        index, v, s, ff, ft, rmf, rmt, rn = job
        progress_queue.put(("message", rn, {"text": f"[{index}/{len(planned)}] start {rn}"}))
        train_model(
            yaml_path=yaml_path,
            variant=v, size=s, run_name=rn,
            epochs=epochs, batch_size=batch_size,
            img_size=img_size, device=device,
            fm_freq=ff, fm_time=ft,
            reg_max_freq=rmf, reg_max_time=rmt,
            cached_train_ds=_cached_train_ds,
            cached_val_ds=_cached_val_ds,
            progress_queue=progress_queue,
            progress_job_id=rn,
            quiet=True,
        )
        return rn

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(run_job, job): job for job in jobs}
        pbar = tqdm(
            total=(epochs or EPOCHS) * len(jobs),
            desc="Sweep epochs", unit="epoch", dynamic_ncols=True,
            file=sys.stdout,
        )
        last_epochs         = {job[_RN]: 0               for job in jobs}
        total_epochs_by_job = {job[_RN]: epochs or EPOCHS for job in jobs}
        active_jobs: set    = set()
        pending             = set(future_to_job)

        try:
            while pending:
                drain_progress(pbar, last_epochs, total_epochs_by_job, active_jobs)
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    rn = future_to_job[future][_RN]
                    try:
                        future.result()
                        completed += 1
                        drain_progress(pbar, last_epochs, total_epochs_by_job, active_jobs)
                        tqdm.write(f"Completed [{completed}/{len(jobs)}]: {rn}")
                    except Exception as exc:
                        failures.append((rn, str(exc)))
                        tqdm.write(f"Failed: {rn}: {exc}")
        finally:
            drain_progress(pbar, last_epochs, total_epochs_by_job, active_jobs)
            pbar.close()

    print("\n" + "=" * 70)
    print("Sweep complete")
    print("=" * 70)
    if failures:
        print(f"Failures: {len(failures)}")
        for rn, err in failures:
            print(f"  {rn}: {err}")
    else:
        print("All requested runs completed.")
    if skipped:
        print(f"Skipped: {len(skipped)}")
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    yaml_path = validate_split_yaml(output_dir=OUTPUT_DIR)

    if RUN_MODE == "train":
        train_model(
            yaml_path=yaml_path,
            variant=VARIANT, size=SIZE, run_name=RUN_NAME,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            img_size=IMG_SIZE, device=DEVICE,
        )
        return

    if RUN_MODE == "sweep":
        sweep_models(
            yaml_path=yaml_path,
            versions=SWEEP_VARIANTS, scales=SWEEP_SCALES,
            fm_freqs=SWEEP_FM_FREQS, fm_times=SWEEP_FM_TIMES,
            coverage_levels=SWEEP_COVERAGE_LEVELS,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            img_size=IMG_SIZE, device=DEVICE,
            parallel=SWEEP_PARALLEL,
            skip_missing_local=SKIP_MISSING_LOCAL,
        )
        return

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE!r}")


if __name__ == "__main__":
    main()
