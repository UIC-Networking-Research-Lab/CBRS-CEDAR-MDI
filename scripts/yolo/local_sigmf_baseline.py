"""
Sequential local baseline for CBRS YOLO models on raw SigMF samples.

Pipeline per sample:
    raw SigMF pair -> image tensor -> full local YOLO forward -> postprocess -> metrics

No distributed nodes, no HTTP transport, no worker threads.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import csv
import gc
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cbrs.sigmf_fast_tensor_preprocessor import FastSigMFTensorPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_baseline_results"
DEFAULT_MODEL_NAME = "cbrs_yolo8x"
DEFAULT_SWEEP_MODELS = [
    "cbrs_yolo8n",
    "cbrs_yolo8s",
    "cbrs_yolo8x",
    "cbrs_yolo11n",
    "cbrs_yolo11s",
    "cbrs_yolo11x",
    "cbrs_yolo26n",
    "cbrs_yolo26s",
    "cbrs_yolo26x",
    "cbrs_rtdetr_l",
]
DEFAULT_SWEEP_DEVICES = ["cpu", "cuda"]
SUPPORTED_MODEL_NAMES = {
    "cbrs_yolo8n": "cbrs_yolo8n",
    "cbrs_yolo8s": "cbrs_yolo8s",
    "cbrs_yolo8x": "cbrs_yolo8x",
    "cbrs_yolo11n": "cbrs_yolo11n",
    "cbrs_yolo11s": "cbrs_yolo11s",
    "cbrs_yolo11x": "cbrs_yolo11x",
    "cbrs_yolo26n": "cbrs_yolo26n",
    "cbrs_yolo26s": "cbrs_yolo26s",
    "cbrs_yolo26x": "cbrs_yolo26x",
    "cbrs_rtdetr_l": "cbrs_rtdetr_l",
}
RAW_SIGMF_ROOT = PROJECT_ROOT / "raw_sigmf_split"
YOLO_SPLIT_ROOT = PROJECT_ROOT / "yolo_split"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES = {0: "LTE", 1: "Radar"}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
        raise ValueError(f"Unsupported model_name={model_name!r}; supported: {supported}")
    return SUPPORTED_MODEL_NAMES[key]


def model_path_for_name(model_name: str) -> Path:
    return PROJECT_ROOT / "trained_models" / model_name / "weights" / "best.pt"


def is_rtdetr_model(model_name: str) -> bool:
    return "rtdetr" in model_name.lower()


def is_yolo26_model(model_name: str) -> bool:
    return "yolo26" in model_name.lower()


def load_detection_model(model_name: str, model_path: Path, device: torch.device):
    try:
        if is_rtdetr_model(model_name):
            from ultralytics import RTDETR

            model = RTDETR(str(model_path))
        else:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to load detection models") from exc

    model.model.to(device)
    model.model.float()
    model.model.eval()
    return model


def cleanup_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def yolo_split_name(raw_split: str) -> str:
    return "val" if raw_split == "validation" else raw_split


def collect_sigmf_pairs(raw_root: Path, split: str, max_samples: int | None) -> list[tuple[Path, Path]]:
    split_dir = raw_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Raw split directory not found: {split_dir}")

    pairs: list[tuple[Path, Path]] = []
    for meta_path in sorted(split_dir.glob("*.sigmf-meta")):
        data_path = split_dir / meta_path.name.replace(".sigmf-meta", ".sigmf-data")
        if data_path.exists():
            pairs.append((meta_path, data_path))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    return pairs


def label_path_for_sample(yolo_root: Path, raw_split: str, sample_id: str) -> Path:
    return yolo_root / yolo_split_name(raw_split) / "labels" / f"{sample_id}.txt"


def load_yolo_label_tensor(label_path: Path) -> torch.Tensor:
    labels = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
            x1 = (x_center - width / 2.0) * IMG_SIZE
            y1 = (y_center - height / 2.0) * IMG_SIZE
            x2 = (x_center + width / 2.0) * IMG_SIZE
            y2 = (y_center + height / 2.0) * IMG_SIZE
            labels.append([cls, x1, y1, x2, y2])
    if not labels:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.float32)


def preprocess_sigmf_to_tensor(
    meta_path: Path,
    data_path: Path,
    preprocessor: FastSigMFTensorPreprocessor,
) -> torch.Tensor:
    return preprocessor.raw_to_tensor(meta_path, data_path)


def tensor_shapes(value: Any, limit: int = 16) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    seen: set[int] = set()

    def walk(item: Any) -> None:
        if len(shapes) >= limit:
            return
        if torch.is_tensor(item):
            item_id = id(item)
            if item_id in seen:
                return
            seen.add(item_id)
            shapes.append(
                {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype).replace("torch.", ""),
                    "bytes": int(item.numel() * item.element_size()),
                }
            )
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    return shapes


def move_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [move_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(v) for v in value)
    return value


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), dtype=torch.float32)

    inter_x1 = torch.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-9)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xyxy = boxes.clone()
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy.clamp(0, IMG_SIZE)


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        iou = box_iou_xyxy(boxes[current].view(1, 4), boxes[order[1:]]).view(-1)
        order = order[1:][iou <= iou_threshold]
    if not keep:
        return torch.zeros((0,), dtype=torch.long)
    return torch.stack(keep).long()


def postprocess_yolo_raw_output(raw_output: Any, conf_threshold: float, iou_threshold: float) -> torch.Tensor:
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    prediction = move_to_cpu(prediction)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.shape[0] == 4 + len(CLASS_NAMES):
        prediction = prediction.T
    if prediction.ndim != 2 or prediction.shape[1] < 4 + len(CLASS_NAMES):
        raise RuntimeError(f"Unsupported YOLO raw output shape: {tuple(prediction.shape)}")

    boxes = xywh_to_xyxy(prediction[:, :4].float())
    class_scores = prediction[:, 4 : 4 + len(CLASS_NAMES)].float()
    scores, classes = class_scores.max(dim=1)
    mask = scores >= float(conf_threshold)
    if not bool(mask.any()):
        return torch.zeros((0, 6), dtype=torch.float32)

    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask].float()
    selected = []
    for class_id in CLASS_NAMES:
        class_mask = classes == float(class_id)
        if not bool(class_mask.any()):
            continue
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        keep = simple_nms(class_boxes, class_scores, float(iou_threshold))
        selected.append(
            torch.cat(
                [
                    class_boxes[keep],
                    class_scores[keep, None],
                    torch.full((keep.numel(), 1), float(class_id)),
                ],
                dim=1,
            )
        )

    if not selected:
        return torch.zeros((0, 6), dtype=torch.float32)
    detections = torch.cat(selected, dim=0)
    return detections[detections[:, 4].argsort(descending=True)[:300]].float()


def postprocess_yolo26_raw_output(raw_output: Any, conf_threshold: float, iou_threshold: float) -> torch.Tensor:
    _ = iou_threshold
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    prediction = move_to_cpu(prediction)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim == 2 and prediction.shape[0] == 6 and prediction.shape[1] != 6:
        prediction = prediction.T
    if prediction.ndim != 2 or prediction.shape[1] < 6:
        raise RuntimeError(f"Unsupported YOLO26 raw output shape: {tuple(prediction.shape)}")

    boxes = prediction[:, :4].float().clamp(0, IMG_SIZE)
    scores = prediction[:, 4].float()
    classes = prediction[:, 5].round().long()
    valid_classes = torch.zeros_like(classes, dtype=torch.bool)
    for class_id in CLASS_NAMES:
        valid_classes |= classes == int(class_id)
    mask = (scores >= float(conf_threshold)) & valid_classes
    if not bool(mask.any()):
        return torch.zeros((0, 6), dtype=torch.float32)

    detections = torch.cat(
        [
            boxes[mask],
            scores[mask, None],
            classes[mask, None].float(),
        ],
        dim=1,
    )
    return detections[detections[:, 4].argsort(descending=True)[:300]].float()


def postprocess_rtdetr_raw_output(raw_output: Any, conf_threshold: float, iou_threshold: float) -> torch.Tensor:
    _ = iou_threshold
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    prediction = move_to_cpu(prediction)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.shape[1] < 4 + len(CLASS_NAMES):
        raise RuntimeError(f"Unsupported RT-DETR raw output shape: {tuple(prediction.shape)}")

    boxes = xywh_to_xyxy(prediction[:, :4].float() * float(IMG_SIZE))
    class_scores = prediction[:, 4 : 4 + len(CLASS_NAMES)].float()
    scores, classes = class_scores.max(dim=1)
    mask = scores >= float(conf_threshold)
    if not bool(mask.any()):
        return torch.zeros((0, 6), dtype=torch.float32)

    detections = torch.cat(
        [
            boxes[mask],
            scores[mask, None],
            classes[mask, None].float(),
        ],
        dim=1,
    )
    return detections[detections[:, 4].argsort(descending=True)[:300]].float()


def postprocess_raw_output(
    model_name: str,
    raw_output: Any,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[torch.Tensor, str]:
    if is_rtdetr_model(model_name):
        return (
            postprocess_rtdetr_raw_output(raw_output, conf_threshold, iou_threshold),
            "local_rtdetr_raw",
        )
    if is_yolo26_model(model_name):
        return (
            postprocess_yolo26_raw_output(raw_output, conf_threshold, iou_threshold),
            "local_yolo26_nms_free",
        )
    return (
        postprocess_yolo_raw_output(raw_output, conf_threshold, iou_threshold),
        "local_yolo_nms",
    )


def detection_correct_matrix(detections: torch.Tensor, labels: torch.Tensor, iou_thresholds: np.ndarray) -> np.ndarray:
    correct = np.zeros((detections.shape[0], len(iou_thresholds)), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct

    ious = box_iou_xyxy(detections[:, :4].float(), labels[:, 1:5].float())
    class_match = detections[:, 5:6].int() == labels[:, 0].int().view(1, -1)
    for threshold_index, threshold in enumerate(iou_thresholds):
        det_indices, label_indices = torch.where((ious >= float(threshold)) & class_match)
        pairs = [
            (int(det_idx), int(label_idx), float(ious[det_idx, label_idx]))
            for det_idx, label_idx in zip(det_indices, label_indices)
        ]
        pairs.sort(key=lambda item: item[2], reverse=True)
        used_detections: set[int] = set()
        used_labels: set[int] = set()
        for det_idx, label_idx, _iou in pairs:
            if det_idx in used_detections or label_idx in used_labels:
                continue
            correct[det_idx, threshold_index] = True
            used_detections.add(det_idx)
            used_labels.add(label_idx)
    return correct


def has_frequency_overlap_xyxy(boxes: torch.Tensor, classes: torch.Tensor) -> bool:
    lte_boxes = boxes[classes.int() == 0]
    radar_boxes = boxes[classes.int() == 1]
    for lte in lte_boxes:
        for radar in radar_boxes:
            if max(float(lte[0]), float(radar[0])) < min(float(lte[2]), float(radar[2])):
                return True
    return False


def overlap_state_from_detections(detections: torch.Tensor) -> bool:
    if detections.shape[0] == 0:
        return False
    return has_frequency_overlap_xyxy(detections[:, :4].float(), detections[:, 5].float())


def overlap_state_from_labels(labels: torch.Tensor) -> bool:
    if labels.shape[0] == 0:
        return False
    return has_frequency_overlap_xyxy(labels[:, 1:5].float(), labels[:, 0].float())


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


class DetectionMetricAccumulator:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.correct: list[np.ndarray] = []
        self.conf: list[np.ndarray] = []
        self.pred_cls: list[np.ndarray] = []
        self.target_cls: list[np.ndarray] = []

    def add(self, detections: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        detections = detections.detach().cpu()
        labels = labels.detach().cpu()
        correct = detection_correct_matrix(detections, labels, self.iouv)
        conf = detections[:, 4].numpy() if detections.shape[0] else np.zeros((0,), dtype=float)
        pred_cls = detections[:, 5].numpy() if detections.shape[0] else np.zeros((0,), dtype=float)
        target_cls = labels[:, 0].numpy() if labels.shape[0] else np.zeros((0,), dtype=float)

        self.correct.append(correct)
        self.conf.append(conf)
        self.pred_cls.append(pred_cls)
        self.target_cls.append(target_cls)

        tp50 = int(correct[:, 0].sum()) if correct.size else 0
        pred_count = int(detections.shape[0])
        gt_count = int(labels.shape[0])
        return {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tp50": tp50,
            "fp50": pred_count - tp50,
            "fn50": gt_count - tp50,
            "precision50": tp50 / pred_count if pred_count > 0 else 0.0,
            "recall50": tp50 / gt_count if gt_count > 0 else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        target_cls = np.concatenate(self.target_cls) if self.target_cls else np.zeros((0,), dtype=float)
        correct = np.concatenate(self.correct, axis=0) if self.correct else np.zeros((0, len(self.iouv)), dtype=bool)
        conf = np.concatenate(self.conf) if self.conf else np.zeros((0,), dtype=float)
        pred_cls = np.concatenate(self.pred_cls) if self.pred_cls else np.zeros((0,), dtype=float)

        total_gt = int(target_cls.shape[0])
        total_pred = int(pred_cls.shape[0])
        tp50 = int(correct[:, 0].sum()) if correct.size else 0
        ap = np.zeros((len(CLASS_NAMES), len(self.iouv)), dtype=float)
        for class_id in CLASS_NAMES:
            class_mask = pred_cls == class_id
            n_gt = int((target_cls == class_id).sum())
            n_pred = int(class_mask.sum())
            if n_gt == 0 or n_pred == 0:
                continue
            class_correct = correct[class_mask][np.argsort(-conf[class_mask])]
            for threshold_index in range(len(self.iouv)):
                tp = class_correct[:, threshold_index].astype(float)
                fp = 1.0 - tp
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                ap[class_id, threshold_index] = compute_ap(
                    tp_cum / (n_gt + 1e-16),
                    tp_cum / (tp_cum + fp_cum + 1e-16),
                )

        return {
            "model": self.model_name,
            "classes": CLASS_NAMES,
            "iou_thresholds": [float(value) for value in self.iouv],
            "image_count": len(self.target_cls),
            "target_count": total_gt,
            "prediction_count": total_pred,
            "precision": float(tp50 / total_pred) if total_pred > 0 else 0.0,
            "recall": float(tp50 / total_gt) if total_gt > 0 else 0.0,
            "map50": float(ap[:, 0].mean()),
            "map50_95": float(ap.mean()),
            "ap50_by_class": {CLASS_NAMES[class_id]: float(ap[class_id, 0]) for class_id in CLASS_NAMES},
        }


class OverlapMetricAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, pred_overlap: bool, gt_overlap: bool) -> dict[str, Any]:
        tp = int(gt_overlap and pred_overlap)
        tn = int((not gt_overlap) and (not pred_overlap))
        fp = int((not gt_overlap) and pred_overlap)
        fn = int(gt_overlap and (not pred_overlap))
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        return {
            "gt_overlap": int(gt_overlap),
            "pred_overlap": int(pred_overlap),
            "overlap_tp": tp,
            "overlap_tn": tn,
            "overlap_fp": fp,
            "overlap_fn": fn,
        }

    def summary(self) -> dict[str, Any]:
        total = self.tp + self.tn + self.fp + self.fn
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "overlap_accuracy": (self.tp + self.tn) / total if total else 0.0,
            "overlap_precision": precision,
            "overlap_recall": recall,
            "overlap_f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        }


def output_stem(args: argparse.Namespace) -> str:
    return f"baseline_FastSigmf_{args.model_name}_{args.run_location}_{str(args.device).replace(':', '_')}"


def resolve_output_paths(args: argparse.Namespace) -> None:
    stem = output_stem(args)
    if args.output_csv is None:
        args.output_csv = args.output_dir / f"{stem}.csv"
    if args.quality_json is None:
        args.quality_json = args.output_dir / f"{stem}_quality.json"
    if args.plot_png is None:
        args.plot_png = args.output_dir / f"{stem}_plot.png"


def result_fieldnames() -> list[str]:
    return [
        "sample_id",
        "read_convert_sec",
        "inference_sec",
        "postprocess_sec",
        "end_to_end_sec",
        "completed_at_sec",
        "raw_output_shapes",
        "postprocess_backend",
        "gt_count",
        "pred_count",
        "tp50",
        "fp50",
        "fn50",
        "precision50",
        "recall50",
        "gt_overlap",
        "pred_overlap",
        "overlap_tp",
        "overlap_tn",
        "overlap_fp",
        "overlap_fn",
        "quality_error",
    ]


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else 0.0


def render_plot(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[baseline][plot] matplotlib unavailable: {exc}")
        return

    latency_labels = ["Read+Convert", "Inference", "Postprocess", "End-to-End"]
    latency_values = [
        mean_numeric(rows, "read_convert_sec"),
        mean_numeric(rows, "inference_sec"),
        mean_numeric(rows, "postprocess_sec"),
        mean_numeric(rows, "end_to_end_sec"),
    ]
    quality_labels = ["Overlap Acc", "Overlap Prec", "Overlap Rec", "mAP50-95"]
    quality_values = [
        float(summary.get("overlap", {}).get("overlap_accuracy", 0.0)),
        float(summary.get("overlap", {}).get("overlap_precision", 0.0)),
        float(summary.get("overlap", {}).get("overlap_recall", 0.0)),
        float(summary.get("detection", {}).get("map50_95", 0.0)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{summary.get('detection', {}).get('model', 'model')} local baseline ({len(rows)} samples)")
    axes[0].bar(latency_labels, latency_values, color="#4C78A8")
    axes[0].set_title("Mean Sequential Latency")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[1].bar(quality_labels, quality_values, color="#E45756")
    axes[1].set_title("Final Quality")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", labelrotation=20)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
        for patch in ax.patches:
            value = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[baseline][plot] summary plot written to {output_path}")


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model = load_detection_model(args.model_name, args.model_path, device)
    preprocessor = FastSigMFTensorPreprocessor()
    pairs = collect_sigmf_pairs(args.raw_root, args.split, args.max_samples)
    print(f"[baseline] model={args.model_name} samples={len(pairs)} device={device}")

    metrics = DetectionMetricAccumulator(args.model_name)
    overlap_metrics = OverlapMetricAccumulator()
    rows: list[dict[str, Any]] = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    experiment_start = time.perf_counter()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        writer.writeheader()
        for meta_path, data_path in pairs:
            sample_id = meta_path.name.replace(".sigmf-meta", "")
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            x = preprocess_sigmf_to_tensor(meta_path, data_path, preprocessor)
            read_convert_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.inference_mode():
                raw_output = model.model(x.to(device))
            inference_sec = time.perf_counter() - t0

            quality_error = ""
            t0 = time.perf_counter()
            try:
                raw_output_cpu = move_to_cpu(raw_output)
                detections, postprocess_backend = postprocess_raw_output(
                    args.model_name,
                    raw_output_cpu,
                    args.conf_threshold,
                    args.iou_threshold,
                )
                labels = load_yolo_label_tensor(label_path_for_sample(args.yolo_root, args.split, sample_id))
                sample_metrics = metrics.add(detections, labels)
                sample_overlap_metrics = overlap_metrics.add(
                    pred_overlap=overlap_state_from_detections(detections),
                    gt_overlap=overlap_state_from_labels(labels),
                )
            except Exception as exc:
                raw_output_cpu = move_to_cpu(raw_output)
                postprocess_backend = "failed"
                quality_error = repr(exc)
                sample_metrics = {
                    "gt_count": None,
                    "pred_count": None,
                    "tp50": None,
                    "fp50": None,
                    "fn50": None,
                    "precision50": None,
                    "recall50": None,
                }
                sample_overlap_metrics = {
                    "gt_overlap": None,
                    "pred_overlap": None,
                    "overlap_tp": None,
                    "overlap_tn": None,
                    "overlap_fp": None,
                    "overlap_fn": None,
                }
            postprocess_sec = time.perf_counter() - t0
            completed_at_sec = time.perf_counter() - experiment_start

            row = {
                "sample_id": sample_id,
                "read_convert_sec": read_convert_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "end_to_end_sec": time.perf_counter() - t_total,
                "completed_at_sec": completed_at_sec,
                "raw_output_shapes": json.dumps(tensor_shapes(raw_output_cpu)),
                "postprocess_backend": postprocess_backend,
                **sample_metrics,
                **sample_overlap_metrics,
                "quality_error": quality_error,
            }
            writer.writerow(row)
            rows.append(row)
            f.flush()
            print(
                f"[baseline] {sample_id} read+convert={read_convert_sec:.4f}s "
                f"inference={inference_sec:.4f}s postprocess={postprocess_sec:.4f}s "
                f"end_to_end={row['end_to_end_sec']:.4f}s completed_at={completed_at_sec:.4f}s"
            )

    summary = {"detection": metrics.summary(), "overlap": overlap_metrics.summary()}
    args.quality_json.parent.mkdir(parents=True, exist_ok=True)
    args.quality_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[baseline] csv written to {args.output_csv}")
    print(f"[baseline] quality summary written to {args.quality_json}")
    render_plot(rows, summary, args.plot_png)
    return {
        "summary": summary,
        "rows": rows,
        "csv_path": str(args.output_csv),
        "quality_json_path": str(args.quality_json),
        "plot_png_path": str(args.plot_png),
    }


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def summarize_run_result(args: argparse.Namespace, result: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    rows = result.get("rows", [])
    summary = result.get("summary", {})
    detection = summary.get("detection", {})
    overlap = summary.get("overlap", {})
    completed = [float(row["completed_at_sec"]) for row in rows if row.get("completed_at_sec") not in (None, "")]
    elapsed = max(completed) if completed else 0.0
    image_count = int(detection.get("image_count", 0) or 0)
    return {
        "model_name": args.model_name,
        "device": args.device,
        "split": args.split,
        "max_samples": args.max_samples,
        "csv_path": result.get("csv_path"),
        "quality_json_path": result.get("quality_json_path"),
        "plot_png_path": result.get("plot_png_path"),
        "image_count": image_count,
        "target_count": detection.get("target_count"),
        "prediction_count": detection.get("prediction_count"),
        "map50": detection.get("map50"),
        "map50_95": detection.get("map50_95"),
        "det_precision": detection.get("precision"),
        "det_recall": detection.get("recall"),
        "TP": overlap.get("TP"),
        "TN": overlap.get("TN"),
        "FP": overlap.get("FP"),
        "FN": overlap.get("FN"),
        "overlap_accuracy": overlap.get("overlap_accuracy"),
        "overlap_precision": overlap.get("overlap_precision"),
        "overlap_recall": overlap.get("overlap_recall"),
        "overlap_f1": overlap.get("overlap_f1"),
        "mean_read_convert_sec": mean_numeric(rows, "read_convert_sec"),
        "mean_inference_sec": mean_numeric(rows, "inference_sec"),
        "mean_postprocess_sec": mean_numeric(rows, "postprocess_sec"),
        "mean_end_to_end_sec": mean_numeric(rows, "end_to_end_sec"),
        "throughput_samples_per_sec": image_count / elapsed if elapsed > 0 else 0.0,
        "status": status,
        "error": error,
    }


def write_sweep_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep_summary.json"
    csv_path = output_dir / "sweep_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else [
        "model_name",
        "device",
        "status",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] summary csv written to {csv_path}")
    print(f"[sweep] summary json written to {json_path}")


def run_sweep(args: argparse.Namespace) -> None:
    models = [resolve_model_name(model_name) for model_name in parse_csv_list(args.sweep_models)]
    devices = parse_csv_list(args.sweep_devices)
    summary_rows: list[dict[str, Any]] = []

    for device_name in devices:
        for model_name in models:
            run_args = deepcopy(args)
            run_args.sweep = False
            run_args.model_name = model_name
            run_args.model_path = model_path_for_name(model_name)
            run_args.device = device_name
            run_args.output_csv = None
            run_args.quality_json = None
            run_args.plot_png = None
            resolve_output_paths(run_args)

            print(f"[sweep] starting model={model_name} device={device_name}")
            cleanup_runtime()
            if device_name.startswith("cuda") and not torch.cuda.is_available():
                result = {"rows": [], "summary": {"detection": {"image_count": 0}, "overlap": {}}}
                summary_rows.append(
                    summarize_run_result(run_args, result, "skipped", "cuda unavailable")
                )
                print(f"[sweep] skipped model={model_name} device={device_name}: cuda unavailable")
                continue

            try:
                result = run_baseline(run_args)
                summary_rows.append(summarize_run_result(run_args, result, "ok"))
            except Exception as exc:
                result = {"rows": [], "summary": {"detection": {"image_count": 0}, "overlap": {}}}
                summary_rows.append(summarize_run_result(run_args, result, "failed", repr(exc)))
                print(f"[sweep] failed model={model_name} device={device_name}: {exc!r}")
            finally:
                cleanup_runtime()
                write_sweep_summary(summary_rows, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential local CBRS YOLO SigMF baseline")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep_models", default=",".join(DEFAULT_SWEEP_MODELS))
    parser.add_argument("--sweep_devices", default=",".join(DEFAULT_SWEEP_DEVICES))
    parser.add_argument("--raw_root", type=Path, default=RAW_SIGMF_ROOT)
    parser.add_argument("--yolo_root", type=Path, default=YOLO_SPLIT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_location", default="local")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--quality_json", type=Path, default=None)
    parser.add_argument("--plot_png", type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou_threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.sweep:
        args.model_name = resolve_model_name(args.model_name)
    if not args.sweep and args.model_path is None:
        args.model_path = model_path_for_name(args.model_name)
    return args


def main() -> None:
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        resolve_output_paths(args)
        run_baseline(args)


if __name__ == "__main__":
    main()
