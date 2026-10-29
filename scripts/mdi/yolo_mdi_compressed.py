"""
Distributed FastSigmf inference over HTTP using fixed module-boundary partitions.

The model split is loaded from configs/module_boundary_partitions.json:
    A: FastSigmf preprocess
    B: backbone
    C: neck or encoder
    D: head or decoder, followed by local postprocess/NMS

Only module-boundary tensors listed in the JSON transfer_layers are sent across
links. A->B uses input_tensor because the preprocessed image is not a model
layer output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import ast
import csv
import gc
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cbrs.compressors import get_compressor
from cbrs.http_transport import MessageType, create_transport
from cbrs.sigmf_fast_tensor_preprocessor import FastSigMFTensorPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTITION_JSON = PROJECT_ROOT / "configs" / "module_boundary_partitions.json"
RAW_SIGMF_ROOT = PROJECT_ROOT / "raw_sigmf_split"
YOLO_SPLIT_ROOT = PROJECT_ROOT / "yolo_split"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "distributed_stage_sweep_results"
DEFAULT_RUN_LOCATION = "jetson"
DEFAULT_SWEEP_RATES_MBPS = "150,50,25"
DEFAULT_MODEL_SWEEP_ORDER = (
    "cbrs_yolo8n",
    "cbrs_yolo8s",
    "cbrs_yolo8x",
    "cbrs_rtdetr_l",
    "cbrs_yolo11n",
    "cbrs_yolo11s",
    "cbrs_yolo11x",
    "cbrs_yolo26n",
    "cbrs_yolo26s",
    "cbrs_yolo26x",
)
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES = {0: "LTE", 1: "Radar"}
NODES = ("A", "B", "C", "D")
WORKER_NODES = ("B", "C", "D")
NEXT_HOP = {"A": "B", "B": "C", "C": "D", "D": "A"}
STAGE_MESSAGE = {
    "B": MessageType.TASK_INPUT,
    "C": MessageType.ENCODER_OUTPUT,
    "D": MessageType.DECODER_STEP,
}
NEXT_MESSAGE = {
    "A": MessageType.TASK_INPUT,
    "B": MessageType.ENCODER_OUTPUT,
    "C": MessageType.DECODER_STEP,
    "D": MessageType.FINAL_RESULT,
}
LINK_INDEX = {
    ("A", "B"): 0,
    ("B", "C"): 1,
    ("C", "D"): 2,
    ("D", "A"): 3,
}
COMPRESSED_EDGE_NAMES = {
    ("A", "B"): "a_to_b",
    ("B", "C"): "b_to_c",
    ("C", "D"): "c_to_d",
}
WIRE_TENSOR_MARKER = "__compressed_tensor__"
QUANTIZATION_PARAM_BY_MODE = {
    "fp16": 0.5,
    "int8": 0.25,
    "int4": 0.125,
    "int2": 0.0625,
}
DEFAULT_COMPRESSION_BY_COMPRESSOR = {
    "topk": {"a_to_b": "1.0", "b_to_c": "0.5", "c_to_d": "0.5"},
    "quantization": {"a_to_b": "fp16", "b_to_c": "int8", "c_to_d": "int8"},
    "quant": {"a_to_b": "fp16", "b_to_c": "int8", "c_to_d": "int8"},
    "llmint8": {
        "a_to_b": "[0.01, 'fp16', 'int8']",
        "b_to_c": "[0.01, 'fp16', 'int8']",
        "c_to_d": "[0.01, 'fp16', 'int8']",
    },
    "outliertopk": {"a_to_b": "[0.01, 1.0]", "b_to_c": "[0.01, 0.5]", "c_to_d": "[0.01, 0.5]"},
}
DEFAULT_COMPRESSION_SWEEP_SPECS = (
    {"label": "quant_int8", "compressor": "quantization", "setting": "int8", "param": 0.25},
    {"label": "quant_int4", "compressor": "quantization", "setting": "int4", "param": 0.125},
    {"label": "topk_0p25", "compressor": "topk", "setting": "0.25", "param": 0.25},
    {"label": "topk_0p5", "compressor": "topk", "setting": "0.5", "param": 0.5},
    {"label": "topk_0p75", "compressor": "topk", "setting": "0.75", "param": 0.75},
    {
        "label": "llmint8_o0p01_fp16_int8",
        "compressor": "llmint8",
        "setting": "o0.01_fp16_int8",
        "param": [0.01, "fp16", "int8"],
    },
)
DEFAULT_NODE_IPS = {
    "A": "192.168.1.20",
    "B": "192.168.1.21",
    "C": "192.168.1.22",
    "D": "192.168.1.23",
}
DEFAULT_NODE_PORTS = {
    "A": 52000,
    "B": 52001,
    "C": 52002,
    "D": 52003,
}


@dataclass(frozen=True)
class NodeContract:
    node_id: str
    model_name: str
    family_name: str
    module_name: str
    layer_start: int | None
    layer_end: int | None
    input_edge: str | None
    output_edge: str | None
    required_input_layers: tuple[int, ...]
    transfer_output_layers: tuple[int, ...]
    layer_count: int
    postprocess_on_node: bool = False


@dataclass(frozen=True)
class CompressionSweepCase:
    label: str
    compressor: str
    setting: str
    param: Any
    enabled: bool = True


def load_partition_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Partition JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def supported_model_names(config: dict[str, Any]) -> set[str]:
    return set(config.get("models", {}).keys())


def resolve_model_name(model_name: str, config: dict[str, Any]) -> str:
    key = str(model_name).strip()
    models = config.get("models", {})
    if key not in models:
        supported = ", ".join(sorted(models))
        raise ValueError(f"Unsupported model_name={model_name!r}; supported: {supported}")
    return key


def resolve_model_names(model_name: str | None, sweep_models: str, config: dict[str, Any]) -> list[str]:
    if model_name:
        return [resolve_model_name(model_name, config)]

    sweep_value = str(sweep_models).strip()
    if sweep_value.lower() == "all":
        models = config.get("models", {})
        ordered = [model_name for model_name in DEFAULT_MODEL_SWEEP_ORDER if model_name in models]
        extras = [model_name for model_name in models if model_name not in set(ordered)]
        return ordered + extras

    model_names = [value.strip() for value in sweep_value.split(",") if value.strip()]
    if not model_names:
        raise ValueError("--sweep_models must be 'all' or a comma-separated model list")
    return [resolve_model_name(value, config) for value in model_names]


def family_for_model(config: dict[str, Any], model_name: str) -> tuple[str, dict[str, Any]]:
    family_name = config["models"][model_name]["family"]
    return family_name, config["families"][family_name]


def build_node_contract(config: dict[str, Any], model_name: str, node_id: str) -> NodeContract:
    family_name, family = family_for_model(config, model_name)
    if node_id == "A":
        return NodeContract(
            node_id="A",
            model_name=model_name,
            family_name=family_name,
            module_name="preprocess",
            layer_start=None,
            layer_end=None,
            input_edge=None,
            output_edge="A_to_B",
            required_input_layers=(),
            transfer_output_layers=(),
            layer_count=int(family["layer_count"]),
        )

    node_modules = family.get("node_modules")
    if not node_modules or node_id not in node_modules:
        raise RuntimeError(f"Partition JSON missing node_modules[{node_id}] for family={family_name}")
    module_name = node_modules[node_id]
    module_spec = family["modules"][module_name]
    layer_start, layer_end = [int(value) for value in module_spec["layer_range"]]

    input_edge = next(
        edge_name for edge_name, edge in family["edges"].items()
        if edge.get("dst") == node_id
    )
    output_edge = next(
        edge_name for edge_name, edge in family["edges"].items()
        if edge.get("src") == node_id
    )

    return NodeContract(
        node_id=node_id,
        model_name=model_name,
        family_name=family_name,
        module_name=module_name,
        layer_start=layer_start,
        layer_end=layer_end,
        input_edge=input_edge,
        output_edge=output_edge,
        required_input_layers=tuple(int(v) for v in family["edges"][input_edge]["transfer_layers"]),
        transfer_output_layers=tuple(int(v) for v in family["edges"][output_edge]["transfer_layers"]),
        layer_count=int(family["layer_count"]),
        postprocess_on_node=bool(module_spec.get("postprocess_on_node", False)),
    )


def model_path_for_name(model_name: str) -> Path:
    return PROJECT_ROOT / "trained_models" / model_name / "weights" / "best.pt"


def is_rtdetr_model(model_name: str) -> bool:
    return "rtdetr" in model_name.lower()


def is_yolo26_model(model_name: str) -> bool:
    return "yolo26" in model_name.lower()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(model_name: str, model_path: Path, device: torch.device):
    try:
        if is_rtdetr_model(model_name):
            from ultralytics import RTDETR

            model = RTDETR(str(model_path))
        else:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to load CBRS models") from exc

    model.model.to(device)
    model.model.float()
    model.model.eval()
    return model


def get_layers(model, expected_layer_count: int, model_name: str) -> list[Any]:
    layers = getattr(model.model, "model", None)
    if layers is None:
        raise RuntimeError("Could not find Ultralytics model.model layers")
    layers = list(layers)
    if len(layers) != expected_layer_count:
        raise RuntimeError(f"{model_name} expected {expected_layer_count} layers, got {len(layers)}")
    return layers


def resolve_network(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, int]]:
    node_ips = dict(DEFAULT_NODE_IPS)
    node_ports = dict(DEFAULT_NODE_PORTS)
    for node_id in NODES:
        ip_value = getattr(args, f"ip_{node_id.lower()}")
        port_value = getattr(args, f"port_{node_id.lower()}")
        if ip_value is not None:
            node_ips[node_id] = str(ip_value)
        if port_value is not None:
            node_ports[node_id] = int(port_value)
    return node_ips, node_ports


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(v) for v in value)
    return value


def tensor_bytes(value: Any) -> int:
    seen: set[int] = set()

    def walk(item: Any) -> int:
        if torch.is_tensor(item):
            item_id = id(item)
            if item_id in seen:
                return 0
            seen.add(item_id)
            return int(item.numel() * item.element_size())
        if isinstance(item, dict):
            return sum(walk(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(walk(v) for v in item)
        return 0

    return walk(value)


def yolo_split_name(raw_split: str) -> str:
    return "val" if raw_split == "validation" else raw_split


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


def move_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: move_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(v) for v in value)
    return value


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
        class_scores_for_id = scores[class_mask]
        keep = simple_nms(class_boxes, class_scores_for_id, float(iou_threshold))
        selected.append(
            torch.cat(
                [
                    class_boxes[keep],
                    class_scores_for_id[keep, None],
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
        return postprocess_rtdetr_raw_output(raw_output, conf_threshold, iou_threshold), "local_rtdetr_raw"
    if is_yolo26_model(model_name):
        return postprocess_yolo26_raw_output(raw_output, conf_threshold, iou_threshold), "local_yolo26_nms_free"
    return postprocess_yolo_raw_output(raw_output, conf_threshold, iou_threshold), "local_yolo_nms"


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


class OverlapMetricAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, pred_overlap: bool, gt_overlap: bool) -> None:
        self.tp += int(gt_overlap and pred_overlap)
        self.tn += int((not gt_overlap) and (not pred_overlap))
        self.fp += int((not gt_overlap) and pred_overlap)
        self.fn += int(gt_overlap and (not pred_overlap))

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


def parse_rate_list(value: str) -> list[float]:
    rates = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not rates:
        raise ValueError("--sweep_rates_mbps must contain at least one rate")
    return rates


def send_with_rate(
    transport,
    src: str,
    dst: str,
    msg_type: MessageType,
    payload: dict[str, Any],
    rate_mbps: float,
    args: argparse.Namespace,
) -> tuple[int, float]:
    return transport.send_dynamic(
        dst,
        msg_type,
        payload,
        link_idx=LINK_INDEX[(src, dst)],
        rate_bps=float(rate_mbps) * 1_000_000.0,
        timeout=args.request_timeout,
    )


def compression_disabled(compressor_name: str | None) -> bool:
    return str(compressor_name or "none").strip().lower() in {"", "none", "off", "false", "0"}


def normalize_compressor_name(compressor_name: str | None) -> str:
    name = str(compressor_name or "none").strip().lower()
    if name == "quant":
        return "quantization"
    if name in {"llm_int8", "llm.int8"}:
        return "llmint8"
    if name in {"outlier_topk", "outlier-topk"}:
        return "outliertopk"
    return name


def parse_compression_param(compressor_name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, int, list, tuple)):
        return value

    text = str(value).strip()
    lowered = text.lower()
    if lowered in QUANTIZATION_PARAM_BY_MODE:
        return QUANTIZATION_PARAM_BY_MODE[lowered]
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        raise ValueError(f"Invalid compression param for {compressor_name}: {value!r}") from None


def safe_name(value: Any) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    return safe or "value"


def default_compression_setting(compressor_name: str, edge_name: str) -> str:
    defaults = DEFAULT_COMPRESSION_BY_COMPRESSOR.get(compressor_name, {})
    return defaults.get(edge_name, "1.0")


def make_uniform_compression_config(case: CompressionSweepCase) -> dict[str, Any]:
    if not case.enabled or compression_disabled(case.compressor):
        return {
            "enabled": False,
            "label": case.label,
            "compressor": "none",
            "setting": case.setting,
            "param": case.param,
            "edges": {},
        }

    get_compressor(case.compressor)
    edges = {
        edge_name: {
            "setting": case.setting,
            "param": case.param,
        }
        for edge_name in ("a_to_b", "b_to_c", "c_to_d")
    }
    return {
        "enabled": True,
        "label": case.label,
        "compressor": case.compressor,
        "setting": case.setting,
        "param": case.param,
        "edges": edges,
    }


def compression_case_from_spec(spec: dict[str, Any]) -> CompressionSweepCase:
    compressor_name = normalize_compressor_name(spec["compressor"])
    setting = str(spec.get("setting", spec.get("param", "")))
    return CompressionSweepCase(
        label=str(spec["label"]),
        compressor=compressor_name,
        setting=setting,
        param=spec["param"],
        enabled=not compression_disabled(compressor_name),
    )


def custom_compression_case(args: argparse.Namespace) -> CompressionSweepCase:
    compressor_name = normalize_compressor_name(args.compressor)
    if compression_disabled(compressor_name):
        return CompressionSweepCase(
            label="nocompress",
            compressor="none",
            setting="none",
            param=1.0,
            enabled=False,
        )

    get_compressor(compressor_name)
    settings = [
        value
        for value in (
            args.compression_param,
            args.a_to_b_compression,
            args.b_to_c_compression,
            args.c_to_d_compression,
        )
        if value is not None
    ]
    unique_settings = {str(value) for value in settings}
    if len(unique_settings) > 1:
        raise ValueError("Compression sweep requires the same setting on all links; use --compression_param.")
    setting = settings[0] if settings else default_compression_setting(compressor_name, "a_to_b")
    param = parse_compression_param(compressor_name, setting)
    label = f"{safe_name(compressor_name)}_{safe_name(setting)}"
    return CompressionSweepCase(
        label=label,
        compressor=compressor_name,
        setting=str(setting),
        param=param,
        enabled=True,
    )


def build_compression_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    compressor_name = normalize_compressor_name(args.compressor)
    if compressor_name in {"sweep", "default", "all"}:
        return [
            make_uniform_compression_config(compression_case_from_spec(spec))
            for spec in DEFAULT_COMPRESSION_SWEEP_SPECS
        ]
    return [make_uniform_compression_config(custom_compression_case(args))]


def compression_output_part(args: argparse.Namespace, config: dict[str, Any] | None = None) -> str:
    if config is None:
        cases = getattr(args, "compression_cases", None)
        if cases:
            if len(cases) == 1:
                config = cases[0]
            else:
                return "compressor_sweep"
        else:
            config = getattr(args, "compression_config", {"enabled": False, "label": "nocompress"})
    if not config.get("enabled"):
        return "nocompress"
    return safe_name(config.get("label") or f"{config['compressor']}_{config.get('setting', '')}")


def empty_wire_stats() -> dict[str, float]:
    return {
        "original_bytes": 0.0,
        "compressed_bytes": 0.0,
        "compress_sec": 0.0,
        "decompress_sec": 0.0,
        "tensor_count": 0.0,
    }


def add_wire_stats(dst: dict[str, float], src: dict[str, float]) -> None:
    for key in dst:
        dst[key] += float(src.get(key, 0.0))


def is_compressed_wire_tensor(value: Any) -> bool:
    return isinstance(value, dict) and value.get(WIRE_TENSOR_MARKER) is True


def compress_tensor_for_wire(tensor: torch.Tensor, compressor_name: str, compression_param: Any) -> tuple[dict[str, Any], dict[str, float]]:
    cpu_tensor = detach_to_cpu(tensor)
    original_bytes = tensor_bytes(cpu_tensor)
    compressor = get_compressor(compressor_name)

    t0 = time.perf_counter()
    compressed = compressor.compress(cpu_tensor, compression_param)
    compress_sec = time.perf_counter() - t0
    compressed_bytes = compressor.get_compressed_size(compressed)

    return (
        {
            WIRE_TENSOR_MARKER: True,
            "compressor": compressor_name,
            "param": compression_param,
            "data": compressed,
        },
        {
            "original_bytes": float(original_bytes),
            "compressed_bytes": float(compressed_bytes),
            "compress_sec": compress_sec,
            "decompress_sec": 0.0,
            "tensor_count": 1.0,
        },
    )


def compress_value_for_wire(value: Any, compressor_name: str, compression_param: Any) -> tuple[Any, dict[str, float]]:
    stats = empty_wire_stats()
    if value is None:
        return None, stats
    if torch.is_tensor(value):
        packed, tensor_stats = compress_tensor_for_wire(value, compressor_name, compression_param)
        add_wire_stats(stats, tensor_stats)
        return packed, stats
    if isinstance(value, dict):
        restored: dict[Any, Any] = {}
        for key, item in value.items():
            packed_item, item_stats = compress_value_for_wire(item, compressor_name, compression_param)
            restored[key] = packed_item
            add_wire_stats(stats, item_stats)
        return restored, stats
    if isinstance(value, list):
        restored_list = []
        for item in value:
            packed_item, item_stats = compress_value_for_wire(item, compressor_name, compression_param)
            restored_list.append(packed_item)
            add_wire_stats(stats, item_stats)
        return restored_list, stats
    if isinstance(value, tuple):
        restored_tuple = []
        for item in value:
            packed_item, item_stats = compress_value_for_wire(item, compressor_name, compression_param)
            restored_tuple.append(packed_item)
            add_wire_stats(stats, item_stats)
        return tuple(restored_tuple), stats
    return value, stats


def decompress_tensor_from_wire(package: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    compressor = get_compressor(str(package["compressor"]))
    t0 = time.perf_counter()
    restored = compressor.decompress(package["data"]).to(device, non_blocking=True)
    decompress_sec = time.perf_counter() - t0
    return restored, {
        "original_bytes": float(package["data"].get("original_bytes", tensor_bytes(restored))),
        "compressed_bytes": float(compressor.get_compressed_size(package["data"])),
        "compress_sec": 0.0,
        "decompress_sec": decompress_sec,
        "tensor_count": 1.0,
    }


def decompress_value_from_wire(value: Any, device: torch.device) -> tuple[Any, dict[str, float]]:
    stats = empty_wire_stats()
    if is_compressed_wire_tensor(value):
        tensor, tensor_stats = decompress_tensor_from_wire(value, device)
        add_wire_stats(stats, tensor_stats)
        return tensor, stats
    if isinstance(value, dict):
        restored: dict[Any, Any] = {}
        for key, item in value.items():
            restored_item, item_stats = decompress_value_from_wire(item, device)
            restored[key] = restored_item
            add_wire_stats(stats, item_stats)
        return restored, stats
    if isinstance(value, list):
        restored_list = []
        for item in value:
            restored_item, item_stats = decompress_value_from_wire(item, device)
            restored_list.append(restored_item)
            add_wire_stats(stats, item_stats)
        return restored_list, stats
    if isinstance(value, tuple):
        restored_tuple = []
        for item in value:
            restored_item, item_stats = decompress_value_from_wire(item, device)
            restored_tuple.append(restored_item)
            add_wire_stats(stats, item_stats)
        return tuple(restored_tuple), stats
    return value, stats


def record_wire_stats(profiles: dict[str, Any], node_id: str, edge_name: str, prefix: str, stats: dict[str, float]) -> None:
    if stats.get("tensor_count", 0.0) <= 0.0:
        return
    node_profile = profiles.setdefault(node_id, {})
    node_profile[f"{edge_name}_{prefix}_original_bytes"] = int(stats["original_bytes"])
    node_profile[f"{edge_name}_{prefix}_compressed_bytes"] = int(stats["compressed_bytes"])
    node_profile[f"{edge_name}_{prefix}_sec"] = float(stats[f"{prefix}_sec"])
    node_profile[f"{edge_name}_{prefix}_tensor_count"] = int(stats["tensor_count"])


def compress_payload_for_edge(payload: dict[str, Any], src: str, dst: str, compression_config: dict[str, Any]) -> dict[str, Any]:
    edge_name = COMPRESSED_EDGE_NAMES.get((src, dst))
    config = compression_config or {"enabled": False}
    if edge_name is None or not config.get("enabled"):
        return payload

    edge_config = config["edges"][edge_name]
    compressor_name = str(config["compressor"])
    compression_param = edge_config["param"]
    profiles = payload.setdefault("profiles", {})

    if edge_name == "a_to_b":
        payload["input_tensor"], stats = compress_value_for_wire(
            payload.get("input_tensor"),
            compressor_name,
            compression_param,
        )
    else:
        payload["tensor_layers"], stats = compress_value_for_wire(
            payload.get("tensor_layers", {}),
            compressor_name,
            compression_param,
        )
    record_wire_stats(profiles, src, edge_name, "compress", stats)
    return payload


def decompress_payload_from_edge(payload: dict[str, Any], src: str | None, dst: str, device: torch.device) -> dict[str, Any]:
    if src is None:
        return payload
    edge_name = COMPRESSED_EDGE_NAMES.get((src, dst))
    if edge_name is None:
        return payload

    profiles = payload.setdefault("profiles", {})
    if edge_name == "a_to_b":
        payload["input_tensor"], stats = decompress_value_from_wire(payload.get("input_tensor"), device)
    else:
        payload["tensor_layers"], stats = decompress_value_from_wire(payload.get("tensor_layers", {}), device)
    record_wire_stats(profiles, dst, edge_name, "decompress", stats)
    return payload


def format_rate_list(rates: list[float]) -> str:
    return "[" + ",".join(format_rate_value(value) for value in rates) + "]mbps"


def format_rate_value(rate_mbps: float) -> str:
    value = float(rate_mbps)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def format_device_name(device_arg: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(device_arg).lower()).strip("_")
    return safe or "auto"


def format_model_name_part(model_names: list[str]) -> str:
    if len(model_names) == 1:
        return model_names[0]
    return "allmodels"


def output_stem(args: argparse.Namespace) -> str:
    rates = parse_rate_list(args.sweep_rates_mbps)
    sample_part = "all" if args.max_samples is None else str(args.max_samples)
    return (
        f"distributedModule_FastSigmf_{format_model_name_part(args.model_names)}_{args.run_location}_"
        f"{format_device_name(args.device)}_{compression_output_part(args)}_"
        f"rates{format_rate_list(rates)}_samples{sample_part}"
    )


def resolve_output_paths(args: argparse.Namespace) -> None:
    stem = output_stem(args)
    if args.quality_json is None:
        args.quality_json = args.output_dir / f"{stem}_quality.json"
    if args.summary_csv is None:
        args.summary_csv = args.output_dir / f"{stem}_summary.csv"


def case_output_stem(
    args: argparse.Namespace,
    compression_config: dict[str, Any],
    model_name: str,
    rate_mbps: float,
) -> str:
    sample_part = "all" if args.max_samples is None else str(args.max_samples)
    return (
        f"distributedModule_FastSigmf_{model_name}_{args.run_location}_"
        f"{format_device_name(args.device)}_{compression_output_part(args, compression_config)}_"
        f"rate{format_rate_value(rate_mbps)}mbps_samples{sample_part}"
    )


def case_csv_path(
    args: argparse.Namespace,
    compression_config: dict[str, Any],
    model_name: str,
    rate_mbps: float,
) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_dir / f"{case_output_stem(args, compression_config, model_name, rate_mbps)}.csv"


def result_fieldnames() -> list[str]:
    return [
        "case_id",
        "task_id",
        "sample_id",
        "model_name",
        "family_name",
        "compression_label",
        "compressor",
        "compression_setting",
        "compression_param",
        "rate_mbps",
        "partition_json",
        "A_module",
        "B_module",
        "C_module",
        "D_module",
        "A_preprocess_sec",
        "B_compute_sec",
        "C_compute_sec",
        "D_compute_sec",
        "D_postprocess_sec",
        "postprocess_backend",
        "detections_tensor_bytes",
        "a_to_b_payload_bytes",
        "a_to_b_send_sec",
        "b_to_c_payload_bytes",
        "b_to_c_send_sec",
        "c_to_d_payload_bytes",
        "c_to_d_send_sec",
        "d_to_a_payload_bytes",
        "d_to_a_send_sec",
        "total_comm_bytes",
        "total_comm_sec",
        "case_prepare_sec",
        "started_at_sec",
        "end_to_end_sec",
        "completed_at_sec",
        "gt_count",
        "pred_count",
        "tp50",
        "fp50",
        "fn50",
        "precision50",
        "recall50",
        "map50",
        "map50_95",
        "pred_overlap",
        "gt_overlap",
        "overlap_TP",
        "overlap_TN",
        "overlap_FP",
        "overlap_FN",
        "overlap_accuracy",
        "overlap_precision",
        "overlap_recall",
        "overlap_f1",
        "quality_error",
    ]


def summary_fieldnames() -> list[str]:
    return [
        "case_id",
        "model_name",
        "family_name",
        "compression_label",
        "compressor",
        "compression_setting",
        "compression_param",
        "rate_mbps",
        "sample_count",
        "case_start_at_sec",
        "case_end_at_sec",
        "case_total_sec",
        "seconds_per_task",
        "throughput_tasks_per_sec",
        "avg_A_preprocess_sec",
        "avg_B_compute_sec",
        "avg_C_compute_sec",
        "avg_D_compute_sec",
        "avg_D_postprocess_sec",
        "avg_a_to_b_send_sec",
        "avg_b_to_c_send_sec",
        "avg_c_to_d_send_sec",
        "avg_d_to_a_send_sec",
        "avg_total_comm_sec",
        "avg_end_to_end_sec",
        "avg_a_to_b_payload_bytes",
        "avg_b_to_c_payload_bytes",
        "avg_c_to_d_payload_bytes",
        "avg_d_to_a_payload_bytes",
        "avg_total_comm_bytes",
        "gt_count",
        "pred_count",
        "precision50",
        "recall50",
        "map50",
        "map50_95",
        "overlap_TP",
        "overlap_TN",
        "overlap_FP",
        "overlap_FN",
        "overlap_accuracy",
        "overlap_precision",
        "overlap_recall",
        "overlap_f1",
        "quality_error_count",
        "detail_csv",
    ]


def mean_row_value(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and row.get(field) != ""]
    if not values:
        return None
    return float(sum(values) / len(values))


def edge_summary(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    family_name, family = family_for_model(config, model_name)
    return {
        "family": family_name,
        "node_modules": family["node_modules"],
        "modules": family["modules"],
        "edges": family["edges"],
    }


def quality_worker(result_queue: queue.Queue, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.quality_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    global_metrics: dict[tuple[str, str], DetectionMetricAccumulator] = {}
    global_overlap: dict[tuple[str, str], OverlapMetricAccumulator] = {}
    cases: dict[tuple[str, str, float], dict[str, Any]] = {}
    writers: dict[tuple[str, str, float], tuple[Any, csv.DictWriter]] = {}
    expected_per_case = getattr(args, "samples_per_group", None)

    def get_case(item: dict[str, Any]) -> dict[str, Any]:
        model_name = item["model_name"]
        rate = float(item["rate_mbps"])
        compression_config = item["compression_config"]
        compression_label = compression_config["label"]
        key = (compression_label, model_name, rate)
        if key not in cases:
            cases[key] = {
                "case_id": item["case_id"],
                "compression_config": compression_config,
                "compression_label": compression_label,
                "model_name": model_name,
                "rate_mbps": rate,
                "sample_count": 0,
                "rows": [],
                "summary_written": False,
                "csv_path": case_csv_path(args, compression_config, model_name, rate),
                "detection": DetectionMetricAccumulator(model_name),
                "overlap": OverlapMetricAccumulator(),
                "quality_errors": [],
            }
        return cases[key]

    def get_writer(case: dict[str, Any]) -> csv.DictWriter:
        key = (case["compression_label"], case["model_name"], float(case["rate_mbps"]))
        if key not in writers:
            path = case["csv_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=result_fieldnames())
            writer.writeheader()
            writers[key] = (f, writer)
        return writers[key][1]

    def make_task_row(
        item: dict[str, Any],
        profiles: dict[str, Any],
        detection_summary: dict[str, Any],
        detection_counts: dict[str, Any],
        overlap_summary: dict[str, Any],
        pred_overlap: bool | None,
        gt_overlap: bool | None,
        quality_error: str,
    ) -> dict[str, Any]:
        model_name = item["model_name"]
        compression_config = item["compression_config"]
        partition_summary = edge_summary(args.partition_config_data, model_name)
        comm_bytes = [
            item.get("a_to_b_payload_bytes"),
            profiles.get("B", {}).get("send_payload_bytes"),
            profiles.get("C", {}).get("send_payload_bytes"),
            profiles.get("D", {}).get("send_payload_bytes"),
        ]
        comm_sec = [
            item.get("a_to_b_send_sec"),
            profiles.get("B", {}).get("send_sec"),
            profiles.get("C", {}).get("send_sec"),
            profiles.get("D", {}).get("send_sec"),
        ]
        return {
            "case_id": item["case_id"],
            "task_id": item["task_id"],
            "sample_id": item["sample_id"],
            "model_name": model_name,
            "family_name": partition_summary["family"],
            "compression_label": compression_config["label"],
            "compressor": compression_config["compressor"],
            "compression_setting": compression_config.get("setting"),
            "compression_param": json.dumps(compression_config.get("param"), separators=(",", ":")),
            "rate_mbps": item["rate_mbps"],
            "partition_json": json.dumps(partition_summary, separators=(",", ":")),
            "A_module": "preprocess",
            "B_module": partition_summary["node_modules"]["B"],
            "C_module": partition_summary["node_modules"]["C"],
            "D_module": partition_summary["node_modules"]["D"],
            "A_preprocess_sec": item["preprocess_sec"],
            "B_compute_sec": profiles.get("B", {}).get("compute_sec"),
            "C_compute_sec": profiles.get("C", {}).get("compute_sec"),
            "D_compute_sec": profiles.get("D", {}).get("compute_sec"),
            "D_postprocess_sec": profiles.get("D", {}).get("postprocess_sec"),
            "postprocess_backend": profiles.get("D", {}).get("postprocess_backend"),
            "detections_tensor_bytes": profiles.get("D", {}).get("detections_tensor_bytes"),
            "a_to_b_payload_bytes": item.get("a_to_b_payload_bytes"),
            "a_to_b_send_sec": item.get("a_to_b_send_sec"),
            "b_to_c_payload_bytes": profiles.get("B", {}).get("send_payload_bytes"),
            "b_to_c_send_sec": profiles.get("B", {}).get("send_sec"),
            "c_to_d_payload_bytes": profiles.get("C", {}).get("send_payload_bytes"),
            "c_to_d_send_sec": profiles.get("C", {}).get("send_sec"),
            "d_to_a_payload_bytes": profiles.get("D", {}).get("send_payload_bytes"),
            "d_to_a_send_sec": profiles.get("D", {}).get("send_sec"),
            "total_comm_bytes": sum(int(v or 0) for v in comm_bytes),
            "total_comm_sec": sum(float(v or 0.0) for v in comm_sec),
            "case_prepare_sec": item["case_prepare_sec"],
            "started_at_sec": item["started_at_sec"],
            "end_to_end_sec": item["end_to_end_sec"],
            "completed_at_sec": item["completed_at_sec"],
            "gt_count": detection_counts.get("gt_count", 0),
            "pred_count": detection_counts.get("pred_count", 0),
            "tp50": detection_counts.get("tp50", 0),
            "fp50": detection_counts.get("fp50", 0),
            "fn50": detection_counts.get("fn50", 0),
            "precision50": detection_counts.get("precision50", 0.0),
            "recall50": detection_counts.get("recall50", 0.0),
            "map50": detection_summary["map50"],
            "map50_95": detection_summary["map50_95"],
            "pred_overlap": pred_overlap,
            "gt_overlap": gt_overlap,
            "overlap_TP": overlap_summary["TP"],
            "overlap_TN": overlap_summary["TN"],
            "overlap_FP": overlap_summary["FP"],
            "overlap_FN": overlap_summary["FN"],
            "overlap_accuracy": overlap_summary["overlap_accuracy"],
            "overlap_precision": overlap_summary["overlap_precision"],
            "overlap_recall": overlap_summary["overlap_recall"],
            "overlap_f1": overlap_summary["overlap_f1"],
            "quality_error": quality_error,
        }

    def write_case_summary(summary_writer: csv.DictWriter, case: dict[str, Any]) -> None:
        rows = case["rows"]
        model_name = case["model_name"]
        rate_mbps = float(case["rate_mbps"])
        compression_config = case["compression_config"]
        partition_summary = edge_summary(args.partition_config_data, model_name)
        detection_summary = case["detection"].summary()
        overlap_summary = case["overlap"].summary()
        sample_count = int(case["sample_count"])
        case_start = 0.0
        case_end = max((float(row["completed_at_sec"]) for row in rows), default=case_start)
        case_total = max(0.0, case_end)
        summary_writer.writerow(
            {
                "case_id": case["case_id"],
                "model_name": model_name,
                "family_name": partition_summary["family"],
                "compression_label": compression_config["label"],
                "compressor": compression_config["compressor"],
                "compression_setting": compression_config.get("setting"),
                "compression_param": json.dumps(compression_config.get("param"), separators=(",", ":")),
                "rate_mbps": rate_mbps,
                "sample_count": sample_count,
                "case_start_at_sec": case_start,
                "case_end_at_sec": case_end,
                "case_total_sec": case_total,
                "seconds_per_task": case_total / sample_count if sample_count else 0.0,
                "throughput_tasks_per_sec": sample_count / case_total if case_total > 0 else 0.0,
                "avg_A_preprocess_sec": mean_row_value(rows, "A_preprocess_sec"),
                "avg_B_compute_sec": mean_row_value(rows, "B_compute_sec"),
                "avg_C_compute_sec": mean_row_value(rows, "C_compute_sec"),
                "avg_D_compute_sec": mean_row_value(rows, "D_compute_sec"),
                "avg_D_postprocess_sec": mean_row_value(rows, "D_postprocess_sec"),
                "avg_a_to_b_send_sec": mean_row_value(rows, "a_to_b_send_sec"),
                "avg_b_to_c_send_sec": mean_row_value(rows, "b_to_c_send_sec"),
                "avg_c_to_d_send_sec": mean_row_value(rows, "c_to_d_send_sec"),
                "avg_d_to_a_send_sec": mean_row_value(rows, "d_to_a_send_sec"),
                "avg_total_comm_sec": mean_row_value(rows, "total_comm_sec"),
                "avg_end_to_end_sec": mean_row_value(rows, "end_to_end_sec"),
                "avg_a_to_b_payload_bytes": mean_row_value(rows, "a_to_b_payload_bytes"),
                "avg_b_to_c_payload_bytes": mean_row_value(rows, "b_to_c_payload_bytes"),
                "avg_c_to_d_payload_bytes": mean_row_value(rows, "c_to_d_payload_bytes"),
                "avg_d_to_a_payload_bytes": mean_row_value(rows, "d_to_a_payload_bytes"),
                "avg_total_comm_bytes": mean_row_value(rows, "total_comm_bytes"),
                "gt_count": detection_summary["target_count"],
                "pred_count": detection_summary["prediction_count"],
                "precision50": detection_summary["precision"],
                "recall50": detection_summary["recall"],
                "map50": detection_summary["map50"],
                "map50_95": detection_summary["map50_95"],
                "overlap_TP": overlap_summary["TP"],
                "overlap_TN": overlap_summary["TN"],
                "overlap_FP": overlap_summary["FP"],
                "overlap_FN": overlap_summary["FN"],
                "overlap_accuracy": overlap_summary["overlap_accuracy"],
                "overlap_precision": overlap_summary["overlap_precision"],
                "overlap_recall": overlap_summary["overlap_recall"],
                "overlap_f1": overlap_summary["overlap_f1"],
                "quality_error_count": len(case["quality_errors"]),
                "detail_csv": str(case["csv_path"]),
            }
        )
        case["summary_written"] = True

    summary_f = args.summary_csv.open("w", newline="", encoding="utf-8")
    summary_writer = csv.DictWriter(summary_f, fieldnames=summary_fieldnames())
    summary_writer.writeheader()
    try:
        while True:
            item = result_queue.get()
            try:
                if item is None:
                    break

                result_payload = item["result_envelope"]["payload"]
                detections = result_payload["detections"]
                profiles = result_payload.get("profiles", {})
                case = get_case(item)
                writer = get_writer(case)
                model_name = item["model_name"]
                global_key = (item["compression_config"]["label"], model_name)
                if global_key not in global_metrics:
                    global_metrics[global_key] = DetectionMetricAccumulator(model_name)
                    global_overlap[global_key] = OverlapMetricAccumulator()

                sample_metrics = DetectionMetricAccumulator(model_name)
                sample_overlap = OverlapMetricAccumulator()
                detection_counts: dict[str, Any] = {}
                pred_overlap: bool | None = None
                gt_overlap: bool | None = None
                quality_error = ""
                try:
                    labels = load_yolo_label_tensor(
                        label_path_for_sample(args.yolo_root, args.split, item["sample_id"])
                    )
                    detection_counts = sample_metrics.add(detections, labels)
                    global_metrics[global_key].add(detections, labels)
                    case["detection"].add(detections, labels)

                    pred_overlap = overlap_state_from_detections(detections)
                    gt_overlap = overlap_state_from_labels(labels)
                    sample_overlap.add(pred_overlap=pred_overlap, gt_overlap=gt_overlap)
                    global_overlap[global_key].add(
                        pred_overlap=pred_overlap,
                        gt_overlap=gt_overlap,
                    )
                    case["overlap"].add(pred_overlap=pred_overlap, gt_overlap=gt_overlap)
                except Exception as exc:
                    quality_error = repr(exc)
                    case["quality_errors"].append(f"{item['sample_id']}: {quality_error}")
                    print(f"[A][quality] failed sample={item['sample_id']}: {quality_error}")

                row = make_task_row(
                    item=item,
                    profiles=profiles,
                    detection_summary=sample_metrics.summary(),
                    detection_counts=detection_counts,
                    overlap_summary=sample_overlap.summary(),
                    pred_overlap=pred_overlap,
                    gt_overlap=gt_overlap,
                    quality_error=quality_error,
                )
                writer.writerow(row)
                writers[
                    (
                        item["compression_config"]["label"],
                        model_name,
                        float(item["rate_mbps"]),
                    )
                ][0].flush()
                case["rows"].append(row)
                case["sample_count"] += 1
                if (
                    expected_per_case is not None
                    and case["sample_count"] >= int(expected_per_case)
                    and not case["summary_written"]
                ):
                    write_case_summary(summary_writer, case)
                    summary_f.flush()
            finally:
                result_queue.task_done()
    finally:
        for f, _writer in writers.values():
            f.close()
        for _key, case in sorted(cases.items()):
            if case["sample_count"] > 0 and not case["summary_written"]:
                write_case_summary(summary_writer, case)
        summary_f.close()

    summary = {
        "model_names": args.model_names,
        "rates_mbps": parse_rate_list(args.sweep_rates_mbps),
        "compression_cases": args.compression_cases,
        "models": {
            f"{compression_label}/{model_name}": {
                "partition": edge_summary(args.partition_config_data, model_name),
                "detection": global_metrics[(compression_label, model_name)].summary(),
                "overlap": global_overlap[(compression_label, model_name)].summary(),
            }
            for compression_label, model_name in sorted(global_metrics)
        },
        "cases": {
            f"{compression_label}/{model_name}@{format_rate_value(rate)}mbps": {
                "csv": str(case["csv_path"]),
                "summary_csv": str(args.summary_csv),
                "sample_count": case["sample_count"],
                "compression": case["compression_config"],
                "detection": case["detection"].summary(),
                "overlap": case["overlap"].summary(),
                "quality_error_count": len(case["quality_errors"]),
                "quality_errors": case["quality_errors"][:20],
            }
            for (compression_label, model_name, rate), case in sorted(cases.items())
        },
    }
    args.quality_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for (_compression_label, _model_name, _rate), case in sorted(cases.items()):
        print(f"[A][quality] csv written to {case['csv_path']}")
    print(f"[A][quality] summary csv written to {args.summary_csv}")
    print(f"[A][quality] summary written to {args.quality_json}")


def resolve_layer_ref(outputs: list[Any], current_index: int, ref: int, current_output: Any) -> Any:
    if ref == -1:
        if current_output is None:
            raise RuntimeError(f"Missing current_output for f=-1 at layer={current_index}")
        return current_output
    index = current_index + ref if ref < 0 else ref
    if index < 0 or index >= len(outputs) or outputs[index] is None:
        raise RuntimeError(f"Missing layer output for ref={ref} resolved={index} at layer={current_index}")
    return outputs[index]


def layer_input(module, outputs: list[Any], current_index: int, current_output: Any) -> Any:
    f = getattr(module, "f", -1)
    if isinstance(f, int):
        return resolve_layer_ref(outputs, current_index, int(f), current_output)
    return [resolve_layer_ref(outputs, current_index, int(ref), current_output) for ref in f]


def forward_contract_module(
    layers: list[Any],
    contract: NodeContract,
    input_tensor: Any,
    tensor_layers: dict[int, Any],
    device: torch.device,
) -> tuple[Any, dict[int, Any], float, dict[str, Any]]:
    if contract.layer_start is None or contract.layer_end is None:
        raise RuntimeError("forward_contract_module called for preprocess contract")

    layer_tensors = {int(k): move_to_device(v, device) for k, v in tensor_layers.items()}
    missing = [idx for idx in contract.required_input_layers if idx not in layer_tensors]
    if missing:
        raise RuntimeError(f"[{contract.node_id}] Missing required input layers: {missing}")

    outputs: list[Any] = [None] * contract.layer_count
    for index, value in layer_tensors.items():
        outputs[index] = value

    if input_tensor is not None:
        current_output = move_to_device(input_tensor, device)
    elif contract.layer_start > 0 and outputs[contract.layer_start - 1] is not None:
        current_output = outputs[contract.layer_start - 1]
    else:
        current_output = None

    t0 = time.perf_counter()
    with torch.inference_mode():
        for index in range(contract.layer_start, contract.layer_end):
            module = layers[index]
            current_output = module(layer_input(module, outputs, index, current_output))
            outputs[index] = current_output
    compute_sec = time.perf_counter() - t0

    transfer_layers = {
        index: detach_to_cpu(outputs[index])
        for index in contract.transfer_output_layers
        if outputs[index] is not None
    }
    missing_outputs = [idx for idx in contract.transfer_output_layers if idx not in transfer_layers]
    if missing_outputs:
        raise RuntimeError(f"[{contract.node_id}] Missing transfer output layers: {missing_outputs}")

    raw_output = detach_to_cpu(current_output)
    profile = {
        "module_name": contract.module_name,
        "compute_sec": compute_sec,
        "layer_start": contract.layer_start,
        "layer_end": contract.layer_end,
        "required_input_layers": list(contract.required_input_layers),
        "transfer_output_layers": list(contract.transfer_output_layers),
        "raw_output_tensor_bytes": tensor_bytes(raw_output),
        "transfer_tensor_bytes": tensor_bytes(transfer_layers),
    }
    return raw_output, transfer_layers, compute_sec, profile


def make_infer_payload(
    task: dict[str, Any],
    input_tensor: Any,
    tensor_layers: dict[int, Any],
    profiles: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cmd": "infer",
        "case_id": task["case_id"],
        "task_id": task["task_id"],
        "sample_id": task["sample_id"],
        "model_name": task["model_name"],
        "rate_mbps": task["rate_mbps"],
        "compression_config": task["compression_config"],
        "input_tensor": input_tensor,
        "tensor_layers": tensor_layers,
        "profiles": profiles,
    }


def run_node_a(args: argparse.Namespace) -> None:
    if args.max_inflight <= 0:
        raise ValueError("--max_inflight must be positive")

    rates = args.rates_mbps
    node_ips, node_ports = resolve_network(args)
    device = resolve_device(args.device)
    preprocessor = FastSigMFTensorPreprocessor()
    pairs = collect_sigmf_pairs(args.raw_root, args.split, args.max_samples)
    args.samples_per_group = len(pairs)
    contract = build_node_contract(args.partition_config_data, args.model_names[0], "A")

    transport = create_transport(
        "A",
        node_ips,
        node_ports,
        request_timeout=args.request_timeout,
        rate_mode="dynamic",
        tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=args.ready_timeout)

    task_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    send_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    result_queue: queue.Queue = queue.Queue(maxsize=args.quality_queue_size)
    error_queue: queue.Queue = queue.Queue()
    inflight_slots = threading.Semaphore(args.max_inflight)
    inflight_lock = threading.Lock()
    inflight: dict[str, dict[str, Any]] = {}
    counters = {"sent": 0, "received": 0}
    send_done_event = threading.Event()
    quality_thread = threading.Thread(target=quality_worker, args=(result_queue, args), daemon=False)

    def prepare_case(compression_config: dict[str, Any], model_name: str, rate_mbps: float) -> dict[str, Any]:
        nonlocal preprocessor
        case_id = f"{compression_config['label']}_{model_name}@{format_rate_value(rate_mbps)}mbps"
        t0 = time.perf_counter()
        preprocessor = FastSigMFTensorPreprocessor()
        transport.reset_dynamic_channels()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        payload = {
            "cmd": "prepare_case",
            "case_id": case_id,
            "model_name": model_name,
            "rate_mbps": rate_mbps,
            "compression_config": compression_config,
        }
        for node_id in WORKER_NODES:
            transport.send(node_id, MessageType.CONTROL, payload, timeout=args.request_timeout)

        ready_nodes: dict[str, dict[str, Any]] = {}
        deadline = time.perf_counter() + args.ready_timeout
        while len(ready_nodes) < len(WORKER_NODES):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                missing = sorted(set(WORKER_NODES) - set(ready_nodes))
                raise TimeoutError(f"Timed out waiting for case prepare ack from {missing}")
            envelope = transport.receive(
                msg_type=MessageType.CONTROL,
                predicate=lambda p: (
                    p.get("cmd") == "case_ready"
                    and p.get("case_id") == case_id
                    and p.get("model_name") == model_name
                    and float(p.get("rate_mbps")) == float(rate_mbps)
                ),
                timeout=remaining,
            )
            src = envelope.get("src")
            if src in WORKER_NODES:
                ready_nodes[src] = envelope["payload"]

        case_started_at = time.perf_counter()
        prepare_sec = case_started_at - t0
        print(
            f"[A] case_ready case={case_id} prepare={prepare_sec:.4f}s "
            f"workers={','.join(sorted(ready_nodes))}"
        )
        return {
            "case_id": case_id,
            "compression_config": compression_config,
            "compression_label": compression_config["label"],
            "model_name": model_name,
            "rate_mbps": rate_mbps,
            "case_started_at": case_started_at,
            "case_prepare_sec": prepare_sec,
            "worker_ready": ready_nodes,
        }

    def producer_worker() -> None:
        def wait_for_group_drain(compression_label: str, model_name: str, rate_mbps: float) -> None:
            while True:
                if not error_queue.empty():
                    return
                with inflight_lock:
                    pending = any(
                        meta.get("compression_label") == compression_label
                        and meta.get("model_name") == model_name
                        and float(meta.get("rate_mbps")) == float(rate_mbps)
                        for meta in inflight.values()
                    )
                if not pending:
                    return
                time.sleep(0.1)

        try:
            for compression_config in args.compression_cases:
                for model_name in args.model_names:
                    for rate_mbps in rates:
                        case_meta = prepare_case(compression_config, model_name, rate_mbps)
                        for meta_path, data_path in pairs:
                            inflight_slots.acquire()
                            task_id = uuid.uuid4().hex
                            sample_id = meta_path.name.replace(".sigmf-meta", "")
                            started_at = time.perf_counter()
                            with inflight_lock:
                                inflight[task_id] = {
                                    "case_id": case_meta["case_id"],
                                    "compression_config": case_meta["compression_config"],
                                    "compression_label": case_meta["compression_label"],
                                    "sample_id": sample_id,
                                    "model_name": model_name,
                                    "started_at": started_at,
                                    "case_started_at": case_meta["case_started_at"],
                                    "case_prepare_sec": case_meta["case_prepare_sec"],
                                    "rate_mbps": rate_mbps,
                                    "send_done": None,
                                    "preprocess_sec": None,
                                    "a_to_b_payload_bytes": None,
                                    "a_to_b_send_sec": None,
                                }
                            try:
                                t0 = time.perf_counter()
                                input_tensor = preprocessor.raw_to_tensor(meta_path, data_path)
                                preprocess_sec = time.perf_counter() - t0
                                profiles = {
                                    "A": {
                                        "module_name": contract.module_name,
                                        "preprocess_sec": preprocess_sec,
                                        "output_tensor_bytes": tensor_bytes(input_tensor),
                                    }
                                }
                                task_queue.put(
                                    {
                                        "task_id": task_id,
                                        "case_id": case_meta["case_id"],
                                        "compression_config": case_meta["compression_config"],
                                        "sample_id": sample_id,
                                        "model_name": model_name,
                                        "rate_mbps": rate_mbps,
                                        "case_started_at": case_meta["case_started_at"],
                                        "case_prepare_sec": case_meta["case_prepare_sec"],
                                        "preprocess_sec": preprocess_sec,
                                        "input_tensor": detach_to_cpu(input_tensor),
                                        "profiles": profiles,
                                    }
                                )
                            except Exception:
                                with inflight_lock:
                                    if inflight.pop(task_id, None) is not None:
                                        inflight_slots.release()
                                raise
                        wait_for_group_drain(compression_config["label"], model_name, rate_mbps)
        except Exception as exc:
            error_queue.put(exc)
        finally:
            task_queue.put(None)

    def enqueue_send_worker() -> None:
        try:
            while True:
                task = task_queue.get()
                try:
                    if task is None:
                        break
                    send_done = threading.Event()
                    with inflight_lock:
                        meta = inflight[task["task_id"]]
                        meta["preprocess_sec"] = task["preprocess_sec"]
                        meta["send_done"] = send_done
                    payload = make_infer_payload(
                        task,
                        input_tensor=task["input_tensor"],
                        tensor_layers={},
                        profiles=task["profiles"],
                    )
                    payload = compress_payload_for_edge(payload, "A", "B", task["compression_config"])
                    send_queue.put(
                        {
                            "task_id": task["task_id"],
                            "sample_id": task["sample_id"],
                            "rate_mbps": task["rate_mbps"],
                            "payload": payload,
                            "send_done": send_done,
                        }
                    )
                finally:
                    task_queue.task_done()
        except Exception as exc:
            error_queue.put(exc)
        finally:
            send_queue.put(None)

    def send_worker() -> None:
        try:
            while True:
                item = send_queue.get()
                try:
                    if item is None:
                        break
                    task_id = item["task_id"]
                    send_done = item["send_done"]
                    try:
                        sent_bytes, send_sec = send_with_rate(
                            transport,
                            "A",
                            "B",
                            MessageType.TASK_INPUT,
                            item["payload"],
                            item["rate_mbps"],
                            args,
                        )
                    except Exception:
                        with inflight_lock:
                            inflight.pop(task_id, None)
                        inflight_slots.release()
                        send_done.set()
                        raise
                    with inflight_lock:
                        meta = inflight[task_id]
                        meta["a_to_b_payload_bytes"] = sent_bytes
                        meta["a_to_b_send_sec"] = send_sec
                        counters["sent"] += 1
                    send_done.set()
                    print(
                        f"[A] dispatched sample={item['sample_id']} rate={item['rate_mbps']}Mbps "
                        f"task_id={task_id}"
                    )
                finally:
                    send_queue.task_done()
        except Exception as exc:
            error_queue.put(exc)
        finally:
            send_done_event.set()

    def receive_worker() -> None:
        waiting_for_pending_since = None
        try:
            while True:
                if not error_queue.empty():
                    break
                with inflight_lock:
                    all_received = send_done_event.is_set() and counters["received"] >= counters["sent"]
                if all_received:
                    break
                try:
                    result_envelope = transport.receive(
                        msg_type=MessageType.FINAL_RESULT,
                        source="D",
                        timeout=1.0,
                    )
                except TimeoutError:
                    with inflight_lock:
                        pending = send_done_event.is_set() and counters["received"] < counters["sent"]
                    if pending:
                        now = time.perf_counter()
                        if waiting_for_pending_since is None:
                            waiting_for_pending_since = now
                        elif now - waiting_for_pending_since > args.result_timeout:
                            raise TimeoutError("Timed out waiting for distributed module results")
                    continue

                result = result_envelope["payload"]
                task_id = result.get("task_id")
                with inflight_lock:
                    meta = inflight.get(task_id)
                if meta is None:
                    print(f"[A] ignored unknown result task_id={task_id}")
                    continue

                result.setdefault("profiles", {}).setdefault("D", {})
                result["profiles"]["D"]["send_payload_bytes"] = int(result_envelope.get("payload_bytes", 0))
                result["profiles"]["D"]["send_sec"] = float(result_envelope.get("transfer_elapsed_sec", 0.0))
                meta["send_done"].wait()
                now = time.perf_counter()
                result_queue.put(
                    {
                        "case_id": meta["case_id"],
                        "compression_config": meta["compression_config"],
                        "task_id": task_id,
                        "sample_id": meta["sample_id"],
                        "model_name": meta["model_name"],
                        "rate_mbps": meta["rate_mbps"],
                        "preprocess_sec": meta["preprocess_sec"],
                        "a_to_b_payload_bytes": meta["a_to_b_payload_bytes"],
                        "a_to_b_send_sec": meta["a_to_b_send_sec"],
                        "case_prepare_sec": meta["case_prepare_sec"],
                        "started_at_sec": meta["started_at"] - meta["case_started_at"],
                        "end_to_end_sec": now - meta["started_at"],
                        "completed_at_sec": now - meta["case_started_at"],
                        "result_envelope": result_envelope,
                    }
                )
                with inflight_lock:
                    inflight.pop(task_id, None)
                    counters["received"] += 1
                inflight_slots.release()
                waiting_for_pending_since = None
        except Exception as exc:
            error_queue.put(exc)

    threads = [
        threading.Thread(target=producer_worker, name="module_producer", daemon=True),
        threading.Thread(target=enqueue_send_worker, name="module_enqueue_send", daemon=False),
        threading.Thread(target=send_worker, name="module_send", daemon=False),
        threading.Thread(target=receive_worker, name="module_receive", daemon=False),
    ]
    quality_thread.start()
    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            thread.join()
        if not error_queue.empty():
            raise error_queue.get()
    finally:
        result_queue.put(None)
        result_queue.join()
        quality_thread.join()
        shutdown_rate = rates[0]
        for node_id in WORKER_NODES:
            try:
                transport.send_dynamic(
                    node_id,
                    STAGE_MESSAGE[node_id],
                    {"cmd": "shutdown"},
                    link_idx=LINK_INDEX[("A", node_id)] if ("A", node_id) in LINK_INDEX else 0,
                    rate_bps=float(shutdown_rate) * 1_000_000.0,
                    timeout=args.request_timeout,
                )
            except Exception:
                pass
        transport.close()


def run_worker(node_id: str, args: argparse.Namespace) -> None:
    if args.max_inflight <= 0:
        raise ValueError("--max_inflight must be positive")

    node_ips, node_ports = resolve_network(args)
    device = resolve_device(args.device)
    transport = create_transport(
        node_id,
        node_ips,
        node_ports,
        request_timeout=args.request_timeout,
        rate_mode="dynamic",
        tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes,
    )
    transport.start()
    print(f"[{node_id}] ready, device={device}, models={','.join(args.model_names)}")

    inference_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    send_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    error_queue: queue.Queue = queue.Queue()
    current_model_name: str | None = None
    current_model: Any | None = None
    current_layers: list[Any] | None = None
    current_contract: NodeContract | None = None
    current_case_id: str | None = None
    current_compression_config: dict[str, Any] | None = None
    runtime_lock = threading.Lock()

    def prepare_runtime_for_case(
        case_id: str,
        model_name: str,
        rate_mbps: float,
        compression_config: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal current_case_id, current_model_name, current_model, current_layers, current_contract, current_compression_config
        t0 = time.perf_counter()
        with runtime_lock:
            transport.reset_dynamic_channels()
            current_case_id = None
            current_model_name = None
            current_model = None
            current_layers = None
            current_contract = None
            current_compression_config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            contract = build_node_contract(args.partition_config_data, model_name, node_id)
            model_path = args.model_path if args.model_path is not None else model_path_for_name(model_name)
            model = load_model(model_name, model_path, device)
            layers = get_layers(model, contract.layer_count, model_name)
            current_case_id = case_id
            current_model_name = model_name
            current_model = model
            current_layers = layers
            current_contract = contract
            current_compression_config = compression_config
        load_sec = time.perf_counter() - t0
        print(
            f"[{node_id}] case_ready case={case_id}, model={model_name}, rate={rate_mbps}Mbps, "
            f"load={load_sec:.4f}s, module={contract.module_name}, "
            f"range={contract.layer_start}:{contract.layer_end}, "
            f"requires={list(contract.required_input_layers)}, sends={list(contract.transfer_output_layers)}"
        )
        return {
            "cmd": "case_ready",
            "case_id": case_id,
            "node": node_id,
            "model_name": model_name,
            "rate_mbps": rate_mbps,
            "compression_config": compression_config,
            "module_name": contract.module_name,
            "layer_start": contract.layer_start,
            "layer_end": contract.layer_end,
            "load_sec": load_sec,
        }

    def receive_worker() -> None:
        try:
            while True:
                if not error_queue.empty():
                    break
                try:
                    envelope = transport.receive(timeout=1.0)
                except TimeoutError:
                    continue
                payload = envelope["payload"]
                if payload.get("cmd") == "shutdown":
                    inference_queue.put(None)
                    break
                if envelope["msg_type"] == MessageType.CONTROL and payload.get("cmd") == "prepare_case":
                    ack_payload = prepare_runtime_for_case(
                        case_id=payload["case_id"],
                        model_name=payload["model_name"],
                        rate_mbps=float(payload["rate_mbps"]),
                        compression_config=payload.get("compression_config", {"enabled": False, "label": "nocompress"}),
                    )
                    transport.send("A", MessageType.CONTROL, ack_payload, timeout=args.request_timeout)
                    continue
                if envelope["msg_type"] != STAGE_MESSAGE[node_id]:
                    continue
                inference_queue.put(envelope)
        except Exception as exc:
            error_queue.put(exc)
            inference_queue.put(None)

    def inference_worker() -> None:
        try:
            while True:
                envelope = inference_queue.get()
                try:
                    if envelope is None:
                        send_queue.put(None)
                        break

                    payload = envelope["payload"]
                    model_name = payload["model_name"]
                    case_id = payload["case_id"]
                    task_id = payload["task_id"]
                    profiles = payload.setdefault("profiles", {})
                    src_node = envelope.get("src")
                    if src_node in NODES:
                        profiles.setdefault(src_node, {})
                        profiles[src_node]["send_payload_bytes"] = int(envelope.get("payload_bytes", 0))
                        profiles[src_node]["send_sec"] = float(envelope.get("transfer_elapsed_sec", 0.0))
                    decompress_payload_from_edge(payload, src_node, node_id, device)

                    with runtime_lock:
                        if (
                            current_case_id != case_id
                            or current_model_name != model_name
                            or current_layers is None
                            or current_contract is None
                            or current_compression_config is None
                        ):
                            raise RuntimeError(
                                f"[{node_id}] received task before case prepare: "
                                f"case={case_id}, model={model_name}, "
                                f"current_case={current_case_id}, current_model={current_model_name}"
                            )
                        layers = current_layers
                        contract = current_contract
                        compression_config = current_compression_config
                        raw_output, transfer_layers, compute_sec, profile = forward_contract_module(
                            layers,
                            contract,
                            payload.get("input_tensor"),
                            payload.get("tensor_layers", {}),
                            device,
                        )
                    profiles[node_id] = {**profiles.get(node_id, {}), **profile}
                    dst = NEXT_HOP[node_id]
                    msg_type = NEXT_MESSAGE[node_id]
                    if node_id == "D":
                        t_post = time.perf_counter()
                        detections, postprocess_backend = postprocess_raw_output(
                            model_name,
                            raw_output,
                            args.conf_threshold,
                            args.iou_threshold,
                        )
                        postprocess_sec = time.perf_counter() - t_post
                        detections = detach_to_cpu(detections)
                        profiles[node_id]["postprocess_sec"] = postprocess_sec
                        profiles[node_id]["postprocess_backend"] = postprocess_backend
                        profiles[node_id]["detections_tensor_bytes"] = tensor_bytes(detections)
                        out_payload = {
                            "cmd": "result",
                            "case_id": case_id,
                            "task_id": task_id,
                            "sample_id": payload["sample_id"],
                            "model_name": model_name,
                            "rate_mbps": payload["rate_mbps"],
                            "compression_config": compression_config,
                            "detections": detections,
                            "profiles": profiles,
                        }
                    else:
                        out_payload = {
                            "cmd": "infer",
                            "case_id": case_id,
                            "task_id": task_id,
                            "sample_id": payload["sample_id"],
                            "model_name": model_name,
                            "rate_mbps": payload["rate_mbps"],
                            "compression_config": compression_config,
                            "input_tensor": None,
                            "tensor_layers": transfer_layers,
                            "profiles": profiles,
                        }
                        out_payload = compress_payload_for_edge(out_payload, node_id, dst, compression_config)
                    send_queue.put(
                        {
                            "task_id": task_id,
                            "dst": dst,
                            "msg_type": msg_type,
                            "payload": out_payload,
                            "rate_mbps": payload["rate_mbps"],
                            "compute_sec": compute_sec,
                        }
                    )
                finally:
                    inference_queue.task_done()
        except Exception as exc:
            error_queue.put(exc)
            send_queue.put(None)

    def send_worker() -> None:
        try:
            while True:
                item = send_queue.get()
                try:
                    if item is None:
                        break
                    _sent_bytes, send_sec = send_with_rate(
                        transport,
                        node_id,
                        item["dst"],
                        item["msg_type"],
                        item["payload"],
                        item["rate_mbps"],
                        args,
                    )
                    print(
                        f"[{node_id}] task_id={item['task_id']} compute={item['compute_sec']:.4f}s "
                        f"send->{item['dst']}={send_sec:.4f}s rate={item['rate_mbps']}Mbps"
                    )
                finally:
                    send_queue.task_done()
        except Exception as exc:
            error_queue.put(exc)

    threads = [
        threading.Thread(target=receive_worker, name=f"{node_id}_receive", daemon=False),
        threading.Thread(target=inference_worker, name=f"{node_id}_inference", daemon=False),
        threading.Thread(target=send_worker, name=f"{node_id}_send", daemon=False),
    ]
    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            thread.join()
        if not error_queue.empty():
            raise error_queue.get()
    finally:
        transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed FastSigmf module-boundary inference over HTTP")
    parser.add_argument("--node", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--ip_a", default=None)
    parser.add_argument("--ip_b", default=None)
    parser.add_argument("--ip_c", default=None)
    parser.add_argument("--ip_d", default=None)
    parser.add_argument("--port_a", type=int, default=None)
    parser.add_argument("--port_b", type=int, default=None)
    parser.add_argument("--port_c", type=int, default=None)
    parser.add_argument("--port_d", type=int, default=None)
    parser.add_argument("--model_name", default=None, help="Run one model. If omitted, sweep --sweep_models.")
    parser.add_argument("--sweep_models", default="all", help="Comma-separated models or 'all'.")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--partition_json", type=Path, default=DEFAULT_PARTITION_JSON)
    parser.add_argument("--raw_root", type=Path, default=RAW_SIGMF_ROOT)
    parser.add_argument("--yolo_root", type=Path, default=YOLO_SPLIT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_location", default=DEFAULT_RUN_LOCATION)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--quality_json", type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou_threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--max_inflight", type=int, default=6)
    parser.add_argument("--quality_queue_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--ready_timeout", type=float, default=180.0)
    parser.add_argument("--result_timeout", type=float, default=300.0)
    parser.add_argument("--tx_bucket_capacity_bytes", type=int, default=64 * 1024)
    parser.add_argument("--sweep_rates_mbps", default=DEFAULT_SWEEP_RATES_MBPS)
    parser.add_argument(
        "--compressor",
        default="sweep",
        help="Activation compressor sweep: sweep/default/all, none, topk, quantization, llmint8, or outliertopk.",
    )
    parser.add_argument("--compression_param", default=None, help="Uniform compression setting for all links.")
    parser.add_argument("--a_to_b_compression", default=None, help="A->B compression setting, e.g. 1.0, fp16, int8, or a Python literal list.")
    parser.add_argument("--b_to_c_compression", default=None, help="B->C compression setting, e.g. 0.5, int8, or a Python literal list.")
    parser.add_argument("--c_to_d_compression", default=None, help="C->D compression setting, e.g. 0.5, int8, or a Python literal list.")
    args = parser.parse_args()

    args.partition_config_data = load_partition_config(args.partition_json)
    args.model_names = resolve_model_names(args.model_name, args.sweep_models, args.partition_config_data)
    args.rates_mbps = parse_rate_list(args.sweep_rates_mbps)
    args.compression_cases = build_compression_cases(args)
    if args.model_path is not None and len(args.model_names) != 1:
        raise ValueError("--model_path can only be used with a single --model_name")
    if args.output_csv is not None and (
        len(args.compression_cases) != 1 or len(args.model_names) != 1 or len(args.rates_mbps) != 1
    ):
        raise ValueError("--output_csv can only be used with one compression case, one model, and one rate")
    args.model_name = args.model_names[0] if len(args.model_names) == 1 else format_model_name_part(args.model_names)
    if args.model_path is None and len(args.model_names) == 1:
        args.model_path = model_path_for_name(args.model_names[0])
    resolve_output_paths(args)
    return args


def main() -> None:
    args = parse_args()
    if args.node == "A":
        run_node_a(args)
    else:
        run_worker(args.node, args)


if __name__ == "__main__":
    main()
