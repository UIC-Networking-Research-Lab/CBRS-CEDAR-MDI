"""
test_cedar_yolo.py

CEDAR-YOLO in the paper is called SigYOLO / AsymYOLO (sig_yolo_* runs) in the code.

Evaluate all trained AsymYOLO (SigYOLO) checkpoints found under trained_models/.
Only depends on cbrs/cedar_yolo.py — no main-folder imports.

Metrics aligned with scripts/mdi/yolo_mdi_compressed.py:
  Detection : mAP50, mAP50-95, precision50, recall50, AP50 per class
  Overlap   : TP/TN/FP/FN, accuracy, precision, recall, F1
              + overlap ratio mean/max/min (overlap_width / min_signal_width)

Outputs:
  asym_yolo_results/
    results.json
    summary.csv
    <run_name>/predictions/*.jpg                   (annotated images, JPEG)
    sweep_plots/<variant><size>/fig{1,2,3}_*.png   (per-model sweep heatmaps)

Usage:
    python scripts/cedar/test_cedar_yolo.py [--data yolo_split/data.yaml] [--split test]
                             [--conf 0.25] [--iou 0.45] [--batch 16]
                             [--device cpu|0|cuda] [--models_dir trained_models]
                             [--out_dir asym_yolo_results] [--no_viz]
                             [--jpeg_quality 75] [--no_sweep_plots]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import csv
import json
import math
import re as _re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml as _yaml
from PIL import Image as _PIL, ImageDraw as _ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cbrs.cedar_yolo import build_asym_yolo


# ─────────────────────────────────────────────────────────────────────────────
# YAML parser
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


def parse_data_yaml(yaml_path: Path) -> Tuple[Path, Path, Path, int, List[str]]:
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
    val_img   = _resolve(base, data.get("val",  data["train"]))
    test_img  = _resolve(base, data.get("test", data.get("val", data["train"])))
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        n = max(raw_names.keys()) + 1
        names = [raw_names.get(i, str(i)) for i in range(n)]
    else:
        names = list(raw_names)
    nc = int(data["nc"]) if "nc" in data else len(names)
    return train_img, val_img, test_img, nc, names


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _label_dir(img_dir: Path) -> Path:
    sibling = img_dir.parent / "labels"
    if sibling.exists():
        return sibling
    return img_dir.parent.parent / "labels" / img_dir.name


def _load_label(lbl_path: Path, img_size: int) -> Tensor:
    """[N, 5] cls x1 y1 x2 y2 in pixels."""
    rows: List[List[float]] = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls       = float(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = max(0.0, (cx - w / 2) * img_size)
                y1 = max(0.0, (cy - h / 2) * img_size)
                x2 = min(float(img_size), (cx + w / 2) * img_size)
                y2 = min(float(img_size), (cy + h / 2) * img_size)
                if x2 > x1 and y2 > y1:
                    rows.append([cls, x1, y1, x2, y2])
    return (
        torch.tensor(rows, dtype=torch.float32)
        if rows else torch.zeros((0, 5), dtype=torch.float32)
    )


class _TestDataset(Dataset):
    def __init__(self, img_dir: Path, img_size: int = 640):
        self.img_size = img_size
        self.lbl_dir  = _label_dir(img_dir)
        self.imgs     = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _EXTS)
        if not self.imgs:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[Path, Tensor, Tensor]:
        img_path = self.imgs[idx]
        img = _PIL.open(img_path).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), _PIL.BILINEAR)
        img_t    = TF.to_tensor(img)
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        lbl_t    = _load_label(lbl_path, self.img_size)
        return img_path, img_t, lbl_t


def _collate(batch):
    paths, imgs, labels = zip(*batch)
    return list(paths), torch.stack(imgs), list(labels)


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────

def box_iou_xyxy(box1: Tensor, box2: Tensor) -> Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), dtype=torch.float32)
    inter_x1 = torch.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    a1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    a2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Detection metrics  (mirrors scripts/mdi/yolo_mdi_compressed.py)
# ─────────────────────────────────────────────────────────────────────────────

def detection_correct_matrix(
    detections: Tensor, labels: Tensor, iouv: np.ndarray
) -> np.ndarray:
    correct = np.zeros((detections.shape[0], len(iouv)), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    ious        = box_iou_xyxy(detections[:, :4].float(), labels[:, 1:5].float())
    class_match = detections[:, 5:6].int() == labels[:, 0].int().view(1, -1)
    for ti, thresh in enumerate(iouv):
        det_idx, lbl_idx = torch.where((ious >= float(thresh)) & class_match)
        pairs = [(int(d), int(l), float(ious[d, l])) for d, l in zip(det_idx, lbl_idx)]
        pairs.sort(key=lambda x: x[2], reverse=True)
        used_det: set[int] = set()
        used_lbl: set[int] = set()
        for d, l, _ in pairs:
            if d in used_det or l in used_lbl:
                continue
            correct[d, ti] = True
            used_det.add(d); used_lbl.add(l)
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class DetectionMetricAccumulator:
    def __init__(self, nc: int, class_names: List[str]) -> None:
        self.nc          = nc
        self.class_names = class_names
        self.iouv        = np.linspace(0.5, 0.95, 10)
        self.correct:    List[np.ndarray] = []
        self.conf:       List[np.ndarray] = []
        self.pred_cls:   List[np.ndarray] = []
        self.target_cls: List[np.ndarray] = []

    def add(self, detections: Tensor, labels: Tensor) -> Dict[str, Any]:
        detections = detections.detach().cpu()
        labels     = labels.detach().cpu()
        correct  = detection_correct_matrix(detections, labels, self.iouv)
        conf     = detections[:, 4].numpy() if detections.shape[0] else np.zeros(0)
        pred_cls = detections[:, 5].numpy() if detections.shape[0] else np.zeros(0)
        tgt_cls  = labels[:, 0].numpy()    if labels.shape[0]     else np.zeros(0)
        self.correct.append(correct)
        self.conf.append(conf)
        self.pred_cls.append(pred_cls)
        self.target_cls.append(tgt_cls)
        tp50   = int(correct[:, 0].sum()) if correct.size else 0
        n_pred = int(detections.shape[0])
        n_gt   = int(labels.shape[0])
        return {
            "gt_count":    n_gt,
            "pred_count":  n_pred,
            "tp50":        tp50,
            "fp50":        n_pred - tp50,
            "fn50":        n_gt   - tp50,
            "precision50": tp50 / n_pred if n_pred > 0 else 0.0,
            "recall50":    tp50 / n_gt   if n_gt   > 0 else 0.0,
        }

    def summary(self) -> Dict[str, Any]:
        tgt_cls  = np.concatenate(self.target_cls) if self.target_cls else np.zeros(0)
        correct  = np.concatenate(self.correct, axis=0) if self.correct \
                   else np.zeros((0, len(self.iouv)), dtype=bool)
        conf     = np.concatenate(self.conf)     if self.conf     else np.zeros(0)
        pred_cls = np.concatenate(self.pred_cls) if self.pred_cls else np.zeros(0)
        total_gt   = int(tgt_cls.shape[0])
        total_pred = int(pred_cls.shape[0])
        tp50       = int(correct[:, 0].sum()) if correct.size else 0
        ap = np.zeros((self.nc, len(self.iouv)), dtype=float)
        for c in range(self.nc):
            cls_mask = pred_cls == float(c)
            n_gt_c   = int((tgt_cls == float(c)).sum())
            if n_gt_c == 0 or int(cls_mask.sum()) == 0:
                continue
            cls_correct = correct[cls_mask][np.argsort(-conf[cls_mask])]
            for ti in range(len(self.iouv)):
                tp = cls_correct[:, ti].astype(float)
                fp = 1.0 - tp
                tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
                ap[c, ti] = compute_ap(
                    tp_cum / (n_gt_c + 1e-16),
                    tp_cum / (tp_cum + fp_cum + 1e-16),
                )
        return {
            "image_count":   len(self.target_cls),
            "target_count":  total_gt,
            "pred_count":    total_pred,
            "precision50":   float(tp50 / total_pred) if total_pred > 0 else 0.0,
            "recall50":      float(tp50 / total_gt)   if total_gt   > 0 else 0.0,
            "map50":         float(ap[:, 0].mean()),
            "map50_95":      float(ap.mean()),
            "ap50_by_class": {self.class_names[c]: float(ap[c, 0]) for c in range(self.nc)},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Overlap metrics  (mirrors scripts/mdi/yolo_mdi_compressed.py)
# ─────────────────────────────────────────────────────────────────────────────

def _freq_overlap_pairs(boxes: Tensor, classes: Tensor) -> List[float]:
    """
    Returns list of overlap_ratio values for all LTE-Radar overlapping pairs.
    overlap_ratio = overlap_width / min(lte_width, radar_width)  [pixel space]
    mirrors test_yolo.py :: check_frequency_overlap  (converted to xyxy pixels)
    """
    lte   = boxes[classes.int() == 0]
    radar = boxes[classes.int() == 1]
    ratios: List[float] = []
    for l in lte:
        lx1, lx2 = float(l[0]), float(l[2])
        lw = lx2 - lx1
        for r in radar:
            rx1, rx2 = float(r[0]), float(r[2])
            ox1 = max(lx1, rx1); ox2 = min(lx2, rx2)
            if ox1 < ox2:
                rw = rx2 - rx1
                min_w = min(lw, rw)
                ratios.append((ox2 - ox1) / min_w if min_w > 0 else 0.0)
    return ratios


def overlap_from_detections(detections: Tensor) -> Tuple[bool, List[float]]:
    """Returns (has_overlap, [ratio, ...])."""
    if detections.shape[0] == 0:
        return False, []
    ratios = _freq_overlap_pairs(detections[:, :4], detections[:, 5])
    return len(ratios) > 0, ratios


def overlap_from_labels(labels: Tensor) -> bool:
    if labels.shape[0] == 0:
        return False
    return len(_freq_overlap_pairs(labels[:, 1:5], labels[:, 0])) > 0


class OverlapMetricAccumulator:
    def __init__(self) -> None:
        self.tp = self.tn = self.fp = self.fn = 0
        self._ratios: List[float] = []

    def add(self, pred: bool, gt: bool, ratios: List[float]) -> None:
        self.tp += int(gt and pred)
        self.tn += int((not gt) and (not pred))
        self.fp += int((not gt) and pred)
        self.fn += int(gt and (not pred))
        self._ratios.extend(ratios)

    def summary(self) -> Dict[str, Any]:
        total = self.tp + self.tn + self.fp + self.fn
        prec  = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec   = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        r = self._ratios
        return {
            "overlap_TP":        self.tp,
            "overlap_TN":        self.tn,
            "overlap_FP":        self.fp,
            "overlap_FN":        self.fn,
            "overlap_accuracy":  (self.tp + self.tn) / total if total else 0.0,
            "overlap_precision": prec,
            "overlap_recall":    rec,
            "overlap_f1":        2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
            "overlap_ratio_mean": float(np.mean(r)) if r else None,
            "overlap_ratio_max":  float(np.max(r))  if r else None,
            "overlap_ratio_min":  float(np.min(r))  if r else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run-name parser
# ─────────────────────────────────────────────────────────────────────────────

_RUN_RE = _re.compile(
    r"sig_yolo_(?P<variant>v\d+)(?P<size>[nsmxl])"
    r"_f(?P<fm_freq>\d+)t(?P<fm_time>\d+)"
    r"_c(?P<cpct>\d+)$"
)


def parse_run_name(run_name: str) -> Optional[dict]:
    m = _RUN_RE.match(run_name)
    if not m:
        return None
    fm_freq  = int(m.group("fm_freq"))
    fm_time  = int(m.group("fm_time"))
    coverage = int(m.group("cpct")) / 100.0
    return {
        "variant":      m.group("variant"),
        "size":         m.group("size"),
        "fm_freq":      fm_freq,
        "fm_time":      fm_time,
        "reg_max_freq": max(1, round(coverage * fm_freq)),
        "reg_max_time": max(1, round(coverage * fm_time)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(models_dir: Path) -> List[Tuple[str, Path]]:
    results = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("sig_yolo_"):
            continue
        weights = d / "weights"
        best = weights / "best.pt"
        last = weights / "last.pt"
        if best.exists():
            results.append((d.name, best))
        elif last.exists():
            results.append((d.name, last))
    return results


def load_checkpoint(ckpt_path: Path) -> Optional[dict]:
    try:
        return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] Failed to load {ckpt_path}: {e}", file=sys.stderr)
        return None


def build_model_from_ckpt(ck: dict, run_name: str = "") -> Optional[Tuple]:
    try:
        variant      = ck.get("variant", "v8")
        size         = ck.get("size", "n")
        nc           = ck.get("nc", 2)
        img_size     = ck.get("img_size", 640)
        fm_freq      = ck.get("fm_freq")
        fm_time      = ck.get("fm_time")
        reg_max_freq = ck.get("reg_max_freq")
        reg_max_time = ck.get("reg_max_time")

        if None in (fm_freq, fm_time, reg_max_freq, reg_max_time):
            parsed = parse_run_name(run_name)
            if parsed:
                variant      = ck.get("variant") or parsed["variant"]
                size         = ck.get("size")    or parsed["size"]
                fm_freq      = parsed["fm_freq"]
                fm_time      = parsed["fm_time"]
                reg_max_freq = parsed["reg_max_freq"]
                reg_max_time = parsed["reg_max_time"]
                print(f"  [info] arch from run name: fm={fm_freq}x{fm_time}"
                      f" reg_max={reg_max_freq}x{reg_max_time}")
            else:
                print(f"  [warn] Cannot determine arch params for {run_name!r}", file=sys.stderr)
                return None

        model = build_asym_yolo(
            variant=variant, size=size, nc=nc,
            fm_freq=fm_freq, fm_time=fm_time,
            reg_max_freq=reg_max_freq, reg_max_time=reg_max_time,
            input_w=img_size, input_h=img_size,
        )
        state = ck.get("ema", ck.get("model"))
        if state is None:
            print("  [warn] No weights in checkpoint.", file=sys.stderr)
            return None
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, img_size, nc
    except Exception as e:
        print(f"  [warn] Failed to build model: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

# per-image record for visualization
_Record = Tuple[Path, Tensor, Tensor]   # (img_path, preds[N,6], gt[M,5])


@torch.no_grad()
def evaluate_model(
    model,
    loader:      DataLoader,
    device:      torch.device,
    conf_thres:  float,
    iou_thres:   float,
    nc:          int,
    class_names: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[_Record]]:
    """
    Returns (det_summary, overlap_summary, records).
    records: list of (img_path, preds[N,6], gt[M,5]) for visualization.
    """
    model.to(device)
    model.eval()
    det_acc     = DetectionMetricAccumulator(nc, class_names)
    overlap_acc = OverlapMetricAccumulator()
    records: List[_Record] = []

    for paths, imgs, labels_list in loader:
        imgs = imgs.to(device)
        preds_batch = model.predict(imgs, conf_thres=conf_thres, iou_thres=iou_thres)
        for img_path, preds, gt in zip(paths, preds_batch, labels_list):
            preds = preds.cpu() if preds.shape[0] else torch.zeros((0, 6))
            gt    = gt.cpu()
            det_acc.add(preds, gt)
            has_ov, ratios = overlap_from_detections(preds)
            overlap_acc.add(pred=has_ov, gt=overlap_from_labels(gt), ratios=ratios)
            records.append((img_path, preds, gt))

    return det_acc.summary(), overlap_acc.summary(), records


# ─────────────────────────────────────────────────────────────────────────────
# Visualization  (PIL — no cv2 dependency)
# mirrors test_yolo.py :: predict_with_overlap visualization
# ─────────────────────────────────────────────────────────────────────────────

_LTE_COLOR   = (0,   100, 255)   # blue
_RADAR_COLOR = (220, 0,   0  )   # red
_OV_COLOR    = (220, 0,   0  )   # red text when overlap
_NO_OV_COLOR = (0,   200, 0  )   # green text when no overlap


def _draw_box(draw: _ImageDraw.ImageDraw, x1: float, y1: float,
              x2: float, y2: float, color: tuple, label: str) -> None:
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    tw, th = 6 * len(label), 12
    draw.rectangle([x1, y1 - th - 2, x1 + tw, y1], fill=color)
    draw.text((x1 + 1, y1 - th - 1), label, fill=(255, 255, 255))


def visualize_predictions(
    records:      List[_Record],
    class_names:  List[str],
    out_dir:      Path,
    img_size:     int,
    jpeg_quality: int = 75,
) -> None:
    """Draw predicted boxes on each image and save to out_dir as JPEG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lte_idx   = class_names.index("LTE")   if "LTE"   in class_names else 0
    radar_idx = class_names.index("Radar") if "Radar" in class_names else 1
    cls_colors = {lte_idx: _LTE_COLOR, radar_idx: _RADAR_COLOR}

    for img_path, preds, _gt in records:
        img = _PIL.open(img_path).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), _PIL.BILINEAR)
        draw = _ImageDraw.Draw(img)

        has_ov = False
        if preds.shape[0] > 0:
            # check overlap
            has_ov, _ = overlap_from_detections(preds)
            for row in preds:
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                conf  = float(row[4])
                cls_i = int(row[5])
                name  = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
                color = cls_colors.get(cls_i, (200, 200, 0))
                _draw_box(draw, x1, y1, x2, y2, color, f"{name} {conf:.2f}")

        status = "OVERLAP" if has_ov else "NO OVERLAP"
        color  = _OV_COLOR if has_ov else _NO_OV_COLOR
        draw.rectangle([5, 5, 5 + 12 * len(status), 25], fill=color)
        draw.text((6, 6), status, fill=(255, 255, 255))

        img.save(
            out_dir / (img_path.stem + ".jpg"),
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "run", "epoch", "params_m", "variant", "size",
    "fm_freq", "fm_time", "reg_max_freq", "reg_max_time", "img_size",
    # detection
    "map50", "map50_95", "precision50", "recall50",
    # per-class AP
    "AP_LTE", "AP_Radar",
    # overlap binary
    "overlap_TP", "overlap_TN", "overlap_FP", "overlap_FN",
    "overlap_accuracy", "overlap_precision", "overlap_recall", "overlap_f1",
    # overlap ratio
    "overlap_ratio_mean", "overlap_ratio_max", "overlap_ratio_min",
    # misc
    "elapsed_s", "ckpt",
]


def _row_to_csv(row: dict, names: List[str]) -> dict:
    ds = row.get("det", {})
    ov = row.get("overlap", {})
    ap = ds.get("ap50_by_class", {})
    d: Dict[str, Any] = {
        "run":          row["run"],
        "epoch":        row.get("epoch"),
        "params_m":     row.get("params_m"),
        "variant":      row.get("variant"),
        "size":         row.get("size"),
        "fm_freq":      row.get("fm_freq"),
        "fm_time":      row.get("fm_time"),
        "reg_max_freq": row.get("reg_max_freq"),
        "reg_max_time": row.get("reg_max_time"),
        "img_size":     row.get("img_size"),
        "map50":        ds.get("map50"),
        "map50_95":     ds.get("map50_95"),
        "precision50":  ds.get("precision50"),
        "recall50":     ds.get("recall50"),
    }
    for n in names:
        d[f"AP_{n}"] = ap.get(n)
    for k in ("overlap_TP", "overlap_TN", "overlap_FP", "overlap_FN",
              "overlap_accuracy", "overlap_precision", "overlap_recall", "overlap_f1",
              "overlap_ratio_mean", "overlap_ratio_max", "overlap_ratio_min"):
        d[k] = ov.get(k)
    d["elapsed_s"] = row.get("elapsed_s")
    d["ckpt"]      = row.get("ckpt")
    return d


def write_csv(all_rows: List[dict], names: List[str], out_path: Path) -> None:
    # Build actual field list: replace AP_LTE / AP_Radar with actual class names
    fields = []
    for f in _CSV_FIELDS:
        if f in ("AP_LTE", "AP_Radar"):
            continue
        fields.append(f)
    ap_fields = [f"AP_{n}" for n in names]
    # Insert AP fields after recall50
    idx = fields.index("recall50") + 1
    fields = fields[:idx] + ap_fields + fields[idx:]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(_row_to_csv(row, names))


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print summary table
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, w: int = 6) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return (" " * w + "n/a")[-w-3:]
    return f"{v:{w}.4f}"


def print_table(rows: List[dict], names: List[str]) -> None:
    W = 34
    h  = f"{'Run':<{W}}"
    h += f"  {'mAP50':>6}  {'mAP50-95':>8}  {'prec50':>6}  {'rec50':>6}"
    for n in names:
        h += f"  {'AP_'+n:>9}"
    h += f"  {'ov_acc':>6}  {'ov_prec':>7}  {'ov_rec':>6}  {'ov_F1':>6}"
    h += f"  {'ov_rat_mn':>9}  {'epoch':>5}  {'params':>7}"
    print(h)
    print("-" * len(h))
    for r in rows:
        ds = r.get("det", {})
        ov = r.get("overlap", {})
        line  = f"{r['run']:<{W}}"
        line += f"  {_fmt(ds.get('map50')):>6}"
        line += f"  {_fmt(ds.get('map50_95')):>8}"
        line += f"  {_fmt(ds.get('precision50')):>6}"
        line += f"  {_fmt(ds.get('recall50')):>6}"
        ap_cls = ds.get("ap50_by_class", {})
        for n in names:
            line += f"  {_fmt(ap_cls.get(n)):>9}"
        line += f"  {_fmt(ov.get('overlap_accuracy')):>6}"
        line += f"  {_fmt(ov.get('overlap_precision')):>7}"
        line += f"  {_fmt(ov.get('overlap_recall')):>6}"
        line += f"  {_fmt(ov.get('overlap_f1')):>6}"
        line += f"  {_fmt(ov.get('overlap_ratio_mean')):>9}"
        line += f"  {r.get('epoch', '?'):>5}"
        pm = r.get("params_m")
        line += f"  {pm:6.2f}M" if pm else "      n/a"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep heatmaps
# Per (variant, size):
#   fig1 = N_metrics × N_coverages grid of (fm_freq × fm_time) heatmaps
#   fig2 = (fm_freq × fm_time) grid of coverage line charts
#   fig3 = N_metrics best-coverage summary heatmaps (annotated with optimal cov)
# All heatmaps share viridis cmap + vmin=0/vmax=1; lines use Okabe–Ito palette.
# ─────────────────────────────────────────────────────────────────────────────

try:
    import copy as _copy
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_SWEEP_METRICS: Dict[str, Dict[str, Any]] = {
    "mAP50":             {"color": "#0072B2", "marker": "o", "label": "mAP@50",             "src": ("det",     "map50")},
    "recall50":          {"color": "#56B4E9", "marker": "s", "label": "Det Recall@50",      "src": ("det",     "recall50")},
    "overlap_f1":        {"color": "#D55E00", "marker": "^", "label": "Overlap F1",         "src": ("overlap", "overlap_f1")},
    "overlap_accuracy":  {"color": "#009E73", "marker": "D", "label": "Overlap Accuracy",   "src": ("overlap", "overlap_accuracy")},
    "overlap_recall":    {"color": "#CC79A7", "marker": "v", "label": "Overlap Recall",     "src": ("overlap", "overlap_recall")},
    "overlap_precision": {"color": "#E69F00", "marker": "*", "label": "Overlap Precision",  "src": ("overlap", "overlap_precision")},
}
_SWEEP_CMAP = "viridis"
_SWEEP_VMIN, _SWEEP_VMAX = 0.0, 1.0


def _sweep_rows_to_records(rows: List[dict]) -> List[Dict[str, Any]]:
    """Flatten all_rows into plot records. Falls back to run-name regex if fields missing."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        det = r.get("det") or {}
        ov  = r.get("overlap") or {}
        if not det:
            continue   # failed run
        variant = r.get("variant") if r.get("variant") not in (None, "?") else None
        size    = r.get("size")    if r.get("size")    not in (None, "?") else None
        fm_freq = r.get("fm_freq")
        fm_time = r.get("fm_time")
        rmf     = r.get("reg_max_freq")
        if None in (variant, size, fm_freq, fm_time, rmf):
            parsed = parse_run_name(r.get("run", ""))
            if not parsed:
                continue
            variant = variant or parsed["variant"]
            size    = size    or parsed["size"]
            fm_freq = fm_freq or parsed["fm_freq"]
            fm_time = fm_time or parsed["fm_time"]
            rmf     = rmf     or parsed["reg_max_freq"]
        entry: Dict[str, Any] = {
            "variant": str(variant).lower(),
            "size":    str(size).lower(),
            "fm_freq": int(fm_freq),
            "fm_time": int(fm_time),
            "cpct":    int(round(float(rmf) / float(fm_freq) * 100)),
        }
        for mkey, meta in _SWEEP_METRICS.items():
            sec, field = meta["src"]
            val = (det if sec == "det" else ov).get(field)
            try:
                fv = float(val)
                entry[mkey] = fv if math.isfinite(fv) else float("nan")
            except (TypeError, ValueError):
                entry[mkey] = float("nan")
        out.append(entry)
    return out


def _sweep_make_cmap():
    cmap = _copy.copy(_plt.get_cmap(_SWEEP_CMAP))
    cmap.set_bad("lightgrey")
    return cmap


def _sweep_text_color(val: float, cmap, norm) -> str:
    if not math.isfinite(val):
        return "black"
    rgba = cmap(norm(val))
    lum  = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.45 else "black"


def _sweep_draw_heatmap(ax, mat, fm_freqs, fm_times, cmap, norm,
                        annotate_cov=None, fontsize: int = 8) -> None:
    ax.imshow(mat, cmap=cmap, vmin=_SWEEP_VMIN, vmax=_SWEEP_VMAX, aspect="auto")
    nf, nt = len(fm_freqs), len(fm_times)
    for fi in range(nf):
        for ti in range(nt):
            val = mat[fi, ti]
            if math.isfinite(val):
                tc = _sweep_text_color(val, cmap, norm)
                if annotate_cov is not None and math.isfinite(annotate_cov[fi, ti]):
                    txt = f"{val:.3f}\n(c{int(annotate_cov[fi, ti])})"
                    fs  = max(fontsize - 1, 6)
                else:
                    txt = f"{val:.3f}"
                    fs  = fontsize
                ax.text(ti, fi, txt, ha="center", va="center",
                        fontsize=fs, color=tc, fontweight="bold", linespacing=1.3)
    ax.set_xticks(range(nt)); ax.set_xticklabels(fm_times, fontsize=8)
    ax.set_yticks(range(nf)); ax.set_yticklabels(fm_freqs, fontsize=8)


def _sweep_shared_cbar(fig, ax_list, cmap, norm, label: str = "Metric value (0–1)") -> None:
    sm = _plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_list, location="right", shrink=0.85, pad=0.02, aspect=30)
    cb.set_label(label, fontsize=10)


def _sweep_figure1(records, fm_freqs, fm_times, coverages, title_prefix):
    metric_keys = list(_SWEEP_METRICS.keys())
    n_m, n_c    = len(metric_keys), len(coverages)
    cmap        = _sweep_make_cmap()
    norm        = _mcolors.Normalize(vmin=_SWEEP_VMIN, vmax=_SWEEP_VMAX)
    nf, nt      = len(fm_freqs), len(fm_times)
    fig, axes = _plt.subplots(
        n_m, n_c, figsize=(n_c * 3.2, n_m * 2.9),
        constrained_layout=True, squeeze=False,
    )
    for ri, mkey in enumerate(metric_keys):
        for ci, cpct in enumerate(coverages):
            ax  = axes[ri, ci]
            mat = np.full((nf, nt), np.nan)
            for e in records:
                if e["cpct"] == cpct:
                    try:
                        fi = fm_freqs.index(e["fm_freq"])
                        ti = fm_times.index(e["fm_time"])
                        mat[fi, ti] = e[mkey]
                    except ValueError:
                        pass
            _sweep_draw_heatmap(ax, mat, fm_freqs, fm_times, cmap, norm)
            if ri == 0:
                ax.set_title(f"Coverage = {cpct}%", fontsize=10, fontweight="bold", pad=6)
            ax.set_ylabel(f"{_SWEEP_METRICS[mkey]['label']}\n\nfm_freq" if ci == 0 else "",
                          fontsize=8)
            ax.set_xlabel("fm_time" if ri == n_m - 1 else "", fontsize=8)
    _sweep_shared_cbar(fig, axes, cmap, norm)
    fig.suptitle(
        f"{title_prefix} — Fig 1: FM Feature Size Impact at Each Coverage Level\n"
        f"Rows: metrics  |  Columns: coverage  |  Each cell: {nf}×{nt} heatmap (fm_freq × fm_time)"
        "\nGrey = no model trained yet",
        fontsize=11, fontweight="bold",
    )
    return fig


def _sweep_figure2(records, fm_freqs, fm_times, coverages, title_prefix):
    nf, nt = len(fm_freqs), len(fm_times)
    fig, axes = _plt.subplots(
        nf, nt, figsize=(max(13, nt * 4.3), max(10, nf * 3.3)),
        constrained_layout=True, squeeze=False,
    )
    x_pad = (coverages[-1] - coverages[0]) * 0.10 if len(coverages) > 1 else 10
    for fi, fm_freq in enumerate(fm_freqs):
        for ti, fm_time in enumerate(fm_times):
            ax = axes[fi, ti]
            pool = {e["cpct"]: e for e in records
                    if e["fm_freq"] == fm_freq and e["fm_time"] == fm_time}
            any_line = False
            for mkey, meta in _SWEEP_METRICS.items():
                xs, ys = [], []
                for cpct in coverages:
                    if cpct in pool and math.isfinite(pool[cpct][mkey]):
                        xs.append(cpct); ys.append(pool[cpct][mkey])
                if xs:
                    ax.plot(xs, ys, color=meta["color"], marker=meta["marker"],
                            linewidth=2.0, markersize=6, label=meta["label"])
                    any_line = True
            if not any_line:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
            ax.set_title(f"fm_freq={fm_freq}, fm_time={fm_time}",
                         fontsize=9, fontweight="bold")
            ax.set_xlim(coverages[0] - x_pad, coverages[-1] + x_pad)
            ax.set_ylim(_SWEEP_VMIN - 0.03, _SWEEP_VMAX + 0.03)
            ax.set_xticks(coverages)
            ax.set_xticklabels([f"{c}%" for c in coverages], fontsize=7)
            ax.grid(True, alpha=0.35, linestyle="--")
            ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":")
            if ti == 0:
                ax.set_ylabel("Metric value", fontsize=8)
            if fi == nf - 1:
                ax.set_xlabel("Coverage  (reg_max = coverage × fm)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9, ncol=1,
                   framealpha=0.92, title="Metrics", title_fontsize=9,
                   bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(
        f"{title_prefix} — Fig 2: Coverage Effect per FM Configuration\n"
        "Each subplot = one (fm_freq, fm_time) pair  |  X: coverage %  |  Y: metric value",
        fontsize=11, fontweight="bold",
    )
    return fig


def _sweep_figure3(records, fm_freqs, fm_times, title_prefix):
    metric_keys = list(_SWEEP_METRICS.keys())
    n_m         = len(metric_keys)
    n_cols      = 3
    n_rows      = math.ceil(n_m / n_cols)
    cmap        = _sweep_make_cmap()
    norm        = _mcolors.Normalize(vmin=_SWEEP_VMIN, vmax=_SWEEP_VMAX)
    nf, nt      = len(fm_freqs), len(fm_times)
    fig, axes = _plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * 3.6),
        constrained_layout=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    for idx, mkey in enumerate(metric_keys):
        ax      = axes_flat[idx]
        mat_val = np.full((nf, nt), np.nan)
        mat_cov = np.full((nf, nt), np.nan)
        for fi, fm_freq in enumerate(fm_freqs):
            for ti, fm_time in enumerate(fm_times):
                candidates = [
                    (e[mkey], e["cpct"]) for e in records
                    if e["fm_freq"] == fm_freq and e["fm_time"] == fm_time
                    and math.isfinite(e[mkey])
                ]
                if candidates:
                    best_val, best_cpct = max(candidates, key=lambda x: x[0])
                    mat_val[fi, ti] = best_val
                    mat_cov[fi, ti] = best_cpct
        _sweep_draw_heatmap(ax, mat_val, fm_freqs, fm_times, cmap, norm,
                            annotate_cov=mat_cov)
        ax.set_title(_SWEEP_METRICS[mkey]["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("fm_time", fontsize=8)
        ax.set_ylabel("fm_freq", fontsize=8)
    for idx in range(n_m, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    _sweep_shared_cbar(fig, list(axes_flat[:n_m]), cmap, norm,
                       label="Best metric value (0–1)")
    fig.suptitle(
        f"{title_prefix} — Fig 3: Best-Coverage Summary\n"
        "Each cell: maximum metric across all coverage levels  |  (cXX) = optimal coverage %"
        "\nGrey = no model trained yet",
        fontsize=11, fontweight="bold",
    )
    return fig


def generate_sweep_plots(all_rows: List[dict], out_dir: Path,
                         quiet: bool = False) -> None:
    """For each (variant, size) group, write fig1/fig2/fig3 to
    out_dir/sweep_plots/<variant><size>/.  Grid dims auto-detected from data.
    quiet=True suppresses per-figure logs (used for incremental in-loop updates)."""
    if not _HAS_MPL:
        if not quiet:
            print("[warn] matplotlib not installed — skipping sweep plots.")
        return
    records = _sweep_rows_to_records(all_rows)
    if not records:
        if not quiet:
            print("[warn] No sweep-shaped records — skipping sweep plots.")
        return

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["variant"], r["size"]), []).append(r)

    plots_root = out_dir / "sweep_plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"\nGenerating sweep plots for {len(groups)} (variant,size) group(s) ...")

    n_written = 0
    for (variant, size), recs in sorted(groups.items()):
        tag       = f"{variant}{size}"
        fm_freqs  = sorted({r["fm_freq"] for r in recs})
        fm_times  = sorted({r["fm_time"] for r in recs})
        coverages = sorted({r["cpct"]    for r in recs})
        if max(len(fm_freqs), len(fm_times), len(coverages)) < 2:
            if not quiet:
                print(f"  [{tag}] only one config — skipping (nothing to sweep).")
            continue
        sub = plots_root / tag
        sub.mkdir(parents=True, exist_ok=True)
        title = f"sig_yolo_{tag}"

        builders = [
            ("fig1_fm_heatmaps",
             lambda r=recs, ff=fm_freqs, ft=fm_times, cv=coverages, t=title:
                 _sweep_figure1(r, ff, ft, cv, t)),
            ("fig2_coverage_effect",
             lambda r=recs, ff=fm_freqs, ft=fm_times, cv=coverages, t=title:
                 _sweep_figure2(r, ff, ft, cv, t)),
            ("fig3_best_coverage",
             lambda r=recs, ff=fm_freqs, ft=fm_times, t=title:
                 _sweep_figure3(r, ff, ft, t)),
        ]
        for name, builder in builders:
            path = sub / f"{name}.png"
            if not quiet:
                print(f"  [{tag}] {name}.png")
            fig = builder()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            _plt.close(fig)
        n_written += 1

    if not quiet:
        print(f"Sweep plots saved -> {plots_root}  ({n_written} group(s) written)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AsymYOLO checkpoints")
    p.add_argument("--data",       default="yolo_split/data.yaml")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--device",     default=None, help="cpu | 0 | cuda")
    p.add_argument("--models_dir", default="trained_models")
    p.add_argument("--models", nargs="+", help="Evaluate only these run names")
    p.add_argument("--out_dir",    default="asym_yolo_results")
    p.add_argument("--workers",    type=int,   default=0)
    p.add_argument("--no_viz",     action="store_true", help="skip visualization")
    p.add_argument("--jpeg_quality", type=int, default=75,
                   help="JPEG quality for prediction images (1-95, lower = smaller)")
    p.add_argument("--no_sweep_plots", action="store_true",
                   help="skip per-(variant,size) sweep heatmap generation")
    return p.parse_args()


def main():
    args = parse_args()

    data_yaml  = (PROJECT_ROOT / args.data).resolve()
    models_dir = (PROJECT_ROOT / args.models_dir).resolve()
    out_dir    = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"Device: {device}")

    train_img, val_img, test_img, nc, names = parse_data_yaml(data_yaml)
    split_map = {"train": train_img, "val": val_img, "test": test_img}
    img_dir = split_map[args.split]
    print(f"Split  : {args.split}  ({img_dir})")
    print(f"Classes: {names}")

    checkpoints = discover_checkpoints(models_dir)
    if args.models:
        checkpoints = [(n,p) for n,p in checkpoints if n in args.models]
        missing = set(args.models) - {n for n,p in checkpoints}
        if missing:
            raise FileNotFoundError(f"Missing requested checkpoints: {sorted(missing)}")
    if not checkpoints:
        print(f"No sig_yolo_* folders found in {models_dir}")
        sys.exit(1)
    print(f"\nFound {len(checkpoints)} checkpoint(s).\n")

    all_rows: List[dict] = []
    json_path = out_dir / "results.json"
    csv_path  = out_dir / "summary.csv"

    def _flush_outputs(quiet: bool = True) -> None:
        """Persist current all_rows to results.json / summary.csv and refresh sweep plots."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"split": args.split, "conf": args.conf, "iou": args.iou, "results": all_rows},
                f, indent=2,
                default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x,
            )
        write_csv(all_rows, names, csv_path)
        if not args.no_sweep_plots:
            generate_sweep_plots(all_rows, out_dir, quiet=quiet)

    for run_name, ckpt_path in checkpoints:
        print(f"[{run_name}]  {ckpt_path.name}")
        ck = load_checkpoint(ckpt_path)
        if ck is None:
            all_rows.append({"run": run_name, "det": {}, "overlap": {}, "error": "load_failed"})
            _flush_outputs()
            continue

        result = build_model_from_ckpt(ck, run_name=run_name)
        if result is None:
            all_rows.append({"run": run_name, "det": {}, "overlap": {}, "error": "build_failed"})
            _flush_outputs()
            continue
        model, img_size, model_nc = result

        ds = _TestDataset(img_dir, img_size=img_size)
        loader = DataLoader(
            ds, batch_size=args.batch, shuffle=False,
            collate_fn=_collate, num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

        t0 = time.time()
        det_summary, overlap_summary, records = evaluate_model(
            model, loader, device, args.conf, args.iou, model_nc, names
        )
        elapsed = time.time() - t0

        # Visualization
        if not args.no_viz:
            viz_dir = out_dir / run_name / "predictions"
            print(f"  Saving {len(records)} annotated images (JPEG q={args.jpeg_quality}) -> {viz_dir}")
            visualize_predictions(records, names, viz_dir, img_size,
                                  jpeg_quality=args.jpeg_quality)

        n_params = sum(p.numel() for p in model.parameters())
        epoch    = ck.get("epoch", -1)

        row = {
            "run":          run_name,
            "ckpt":         str(ckpt_path),
            "epoch":        int(epoch) + 1,
            "det":          det_summary,
            "overlap":      overlap_summary,
            "params_m":     n_params / 1e6,
            "elapsed_s":    elapsed,
            "variant":      ck.get("variant", "?"),
            "size":         ck.get("size", "?"),
            "fm_freq":      ck.get("fm_freq"),
            "fm_time":      ck.get("fm_time"),
            "reg_max_freq": ck.get("reg_max_freq"),
            "reg_max_time": ck.get("reg_max_time"),
            "img_size":     img_size,
        }
        all_rows.append(row)

        ap_str = "  ".join(
            f"{n}={det_summary['ap50_by_class'].get(n, float('nan')):.4f}"
            for n in names
        )
        print(
            f"  mAP50={det_summary['map50']:.4f}  mAP50-95={det_summary['map50_95']:.4f}"
            f"  prec={det_summary['precision50']:.4f}  rec={det_summary['recall50']:.4f}"
            f"  {ap_str}"
        )
        ov = overlap_summary
        print(
            f"  overlap: acc={ov['overlap_accuracy']:.4f}  prec={ov['overlap_precision']:.4f}"
            f"  rec={ov['overlap_recall']:.4f}  F1={ov['overlap_f1']:.4f}"
            f"  TP={ov['overlap_TP']} TN={ov['overlap_TN']}"
            f" FP={ov['overlap_FP']} FN={ov['overlap_FN']}"
        )
        ratio_mn = ov.get("overlap_ratio_mean")
        ratio_mx = ov.get("overlap_ratio_max")
        ratio_mi = ov.get("overlap_ratio_min")
        print(
            f"  overlap ratio: mean={ratio_mn:.4f}  max={ratio_mx:.4f}  min={ratio_mi:.4f}"
            if ratio_mn is not None else "  overlap ratio: n/a"
        )
        print(f"  epoch={row['epoch']}  {elapsed:.1f}s\n")

        # per-model JSON
        model_json = out_dir / run_name / "evaluation_summary.json"
        model_json.parent.mkdir(parents=True, exist_ok=True)
        with open(model_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)

        # Incremental flush — results.json / summary.csv / sweep plots refreshed
        # after every model so partial runs are inspectable in real time.
        _flush_outputs(quiet=True)

    # Summary table
    print("\n" + "=" * 130)
    print("Summary")
    print("=" * 130)
    print_table(all_rows, names)

    valid = [r for r in all_rows if r.get("det", {}).get("map50") is not None]
    if valid:
        best = max(valid, key=lambda r: r["det"].get("map50", 0.0))
        print(f"\nBest mAP50: {best['run']}  {best['det']['map50']:.4f}")

    # Final flush (verbose) — defensive for the no-checkpoint case and
    # gives the user a complete plot-generation log for the final state.
    _flush_outputs(quiet=False)

    print(f"\nResults saved -> {out_dir}")
    print(f"  JSON : {json_path}")
    print(f"  CSV  : {csv_path}")
    if _HAS_MPL and not args.no_sweep_plots:
        print(f"  Plots: {out_dir / 'sweep_plots'}")


if __name__ == "__main__":
    main()
