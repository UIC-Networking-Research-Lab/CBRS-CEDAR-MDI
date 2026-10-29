"""
Distributed AsymYOLO FastSigMF inference over HTTP using JSON stage partitions.

CEDAR-YOLO in the paper is called SigYOLO / AsymYOLO (sig_yolo_* runs) in the code.

The transport/message flow mirrors scripts/mdi/yolo_mdi_compressed.py:
    A -> B -> C -> D -> A

By default, partition cases are read from the offline Jetson ranking files in
configs/asym_jetson_partition_rankings_*.json. Passing --partition_json keeps
the old single-partition behavior.
Unlike standard YOLO, AsymYOLO transfers one ordered boundary payload per link;
the tensor_layers field remains present and empty for protocol compatibility.
Preprocess-internal edges are never compressed. image_tensor and model
activation edges are compressed only when their JSON edge policy allows it.
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
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

from cbrs.cedar_yolo import asym_non_max_suppression, build_asym_yolo
from cbrs.compressors import get_compressor
from cbrs.http_transport import MessageType, create_transport
from cbrs.sigmf_fast_tensor_preprocessor import (
    FastSigMFTensorPreprocessor,
    tensor_from_rgb_array,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOCOMPRESS_PARTITION_RANKINGS_JSON = PROJECT_ROOT / "configs" / "asym_jetson_partition_rankings_nocompress.json"
DEFAULT_COMPRESSED_PARTITION_RANKINGS_JSON = PROJECT_ROOT / "configs" / "asym_jetson_partition_rankings_compressed.json"
DEFAULT_PARTITION_TOP_K = 3
PARTITION_FAMILY_NAME = "asym_yolo_v8_f20t20_c100_jetson_comm_aware"
RAW_SIGMF_ROOT = PROJECT_ROOT / "raw_sigmf_split"
YOLO_SPLIT_ROOT = PROJECT_ROOT / "yolo_split"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "distributed_stage_sweep_results_asym"
# The paper's Jetson MDI runs use the earlier CEDAR-YOLO checkpoints (see docs/weights.md).
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models_mdi"
DEFAULT_RUN_LOCATION = "jetson"
DEFAULT_SWEEP_RATES_MBPS = "150,50,25"
DEFAULT_SWEEP_MODELS = (
    "sig_yolo_v8n_f20t20_c100",
    "sig_yolo_v8s_f20t20_c100",
    "sig_yolo_v8x_f20t20_c100",
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
LINK_INDEX = {("A", "B"): 0, ("B", "C"): 1, ("C", "D"): 2, ("D", "A"): 3}
EDGE_NAMES = {("A", "B"): "a_to_b", ("B", "C"): "b_to_c", ("C", "D"): "c_to_d"}
WIRE_TENSOR_MARKER = "__compressed_tensor__"
MODEL_ACTIVATION_PAYLOAD_TYPES = {"model_sppf_output"}
DISALLOWED_BOUNDARY_PAYLOAD_TYPES = {"model_psa_output", "decoded_prediction"}
QUANTIZATION_PARAM_BY_MODE = {"fp16": 0.5, "int8": 0.25, "int4": 0.125, "int2": 0.0625}
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
DEFAULT_NODE_IPS = {"A": "192.168.1.20", "B": "192.168.1.21", "C": "192.168.1.22", "D": "192.168.1.23"}
DEFAULT_NODE_PORTS = {"A": 52000, "B": 52001, "C": 52002, "D": 52003}
_RUN_RE = re.compile(
    r"sig_yolo_(?P<variant>v\d+)(?P<size>[nsmxl])_f(?P<fm_freq>\d+)t(?P<fm_time>\d+)_c(?P<cpct>\d+)$"
)


@dataclass(frozen=True)
class NodeContract:
    node_id: str
    model_name: str
    family_name: str
    module_name: str
    stages: tuple[str, ...]
    input_edge: str | None
    output_edge: str | None
    input_payload_type: str | None
    output_payload_type: str | None
    postprocess_on_node: bool = False


@dataclass(frozen=True)
class CompressionSweepCase:
    label: str
    compressor: str
    setting: str
    param: Any
    enabled: bool = True


def load_partition_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_model_plan(config: dict[str, Any], model_name: str) -> tuple[str, dict[str, Any]]:
    for family_name, family in config.get("families", {}).items():
        plan = family.get("models", {}).get(model_name)
        if plan is not None:
            return family_name, plan
    raise ValueError(f"Unsupported AsymYOLO model_name={model_name!r}")


def validate_partition_boundaries(config: dict[str, Any], source: str = "partition config") -> None:
    for family_name, family in config.get("families", {}).items():
        for model_name, plan in family.get("models", {}).items():
            for edge_name, edge in plan.get("edges", {}).items():
                payload_type = edge.get("payload")
                if payload_type in DISALLOWED_BOUNDARY_PAYLOAD_TYPES:
                    raise ValueError(
                        f"{source} has unsupported AsymYOLO PSA/head boundary: "
                        f"family={family_name}, model={model_name}, edge={edge_name}, payload={payload_type}. "
                        "Regenerate partition rankings with PSA/head marked uncuttable."
                    )


def supported_model_names(config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for family in config.get("families", {}).values():
        names.update(family.get("models", {}).keys())
    return names


def resolve_model_names(model_name: str | None, sweep_models: str, supported: set[str]) -> list[str]:
    if model_name:
        names = [model_name]
    elif str(sweep_models).strip().lower() == "all":
        names = [name for name in DEFAULT_SWEEP_MODELS if name in supported]
    else:
        names = [value.strip() for value in str(sweep_models).split(",") if value.strip()]
    missing = [name for name in names if name not in supported]
    if missing:
        raise ValueError(f"Unsupported AsymYOLO model(s): {missing}; supported={sorted(supported)}")
    return names


def ranking_model_names(*ranking_docs: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for doc in ranking_docs:
        names.update(doc.get("rankings", {}).keys())
    return names


def rate_key(rate_mbps: float) -> str:
    return str(float(rate_mbps))


def partition_id_from_candidate(candidate: dict[str, Any]) -> str:
    rank = int(candidate.get("rank", 0))
    cuts = "_".join(str(int(value)) for value in candidate.get("cuts", []))
    return f"rank{rank:02d}_cuts{cuts}"


def candidate_partition_config(model_name: str, candidate: dict[str, Any]) -> dict[str, Any]:
    plan = dict(candidate)
    return {
        "families": {
            PARTITION_FAMILY_NAME: {
                "graph": "ordered asym yolo stages with communication-aware contiguous chain partition",
                "models": {model_name: plan},
            }
        }
    }


def make_partition_sweep_case(
    model_name: str,
    rate_mbps: float,
    compression_config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    partition_id = partition_id_from_candidate(candidate)
    return {
        "partition_id": partition_id,
        "partition_rank": int(candidate.get("rank", 0)),
        "partition_cuts": list(candidate.get("cuts", [])),
        "partition_config_data": candidate_partition_config(model_name, candidate),
        "model_name": model_name,
        "rate_mbps": float(rate_mbps),
        "compression_config": compression_config,
    }


def build_single_partition_sweep_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for compression_config in args.compression_cases:
        for model_name in args.model_names:
            for rate_mbps in args.rates_mbps:
                family_name, plan = find_model_plan(args.partition_config_data, model_name)
                cuts = plan.get("cuts", [])
                cases.append(
                    {
                        "partition_id": "single_partition",
                        "partition_rank": 1,
                        "partition_cuts": list(cuts),
                        "partition_config_data": {
                            "families": {
                                family_name: {
                                    "graph": "single user-provided asym yolo partition",
                                    "models": {model_name: plan},
                                }
                            }
                        },
                        "model_name": model_name,
                        "rate_mbps": float(rate_mbps),
                        "compression_config": compression_config,
                    }
                )
    return cases


def ranking_candidates_for_case(
    nocompress_doc: dict[str, Any],
    compressed_doc: dict[str, Any],
    model_name: str,
    rate_mbps: float,
    compression_label: str,
) -> list[dict[str, Any]]:
    rate = rate_key(rate_mbps)
    if compression_label == "nocompress":
        block = nocompress_doc.get("rankings", {}).get(model_name, {}).get(rate)
        if block is None:
            raise ValueError(f"Missing nocompress partition rankings for model={model_name}, rate={rate}")
        return list(block.get("top_partitions", []))
    block = compressed_doc.get("rankings", {}).get(model_name, {}).get(rate, {}).get(compression_label)
    if block is None:
        available = sorted(compressed_doc.get("rankings", {}).get(model_name, {}).get(rate, {}).keys())
        raise ValueError(
            f"Missing compressed partition rankings for model={model_name}, rate={rate}, "
            f"compression={compression_label}; available={available}"
        )
    return list(block.get("top_partitions", []))


def build_ranked_partition_sweep_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    nocompress_doc = args.partition_rankings_nocompress_data
    compressed_doc = args.partition_rankings_compressed_data
    cases: list[dict[str, Any]] = []
    for compression_config in args.compression_cases:
        label = compression_config["label"]
        for model_name in args.model_names:
            for rate_mbps in args.rates_mbps:
                candidates = ranking_candidates_for_case(nocompress_doc, compressed_doc, model_name, rate_mbps, label)
                selected = candidates[: args.partition_top_k]
                if not selected:
                    raise ValueError(f"No partition candidates for model={model_name}, rate={rate_mbps}, compression={label}")
                cases.extend(make_partition_sweep_case(model_name, rate_mbps, compression_config, candidate) for candidate in selected)
    return cases


def build_partition_sweep_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.partition_config_data is not None:
        return build_single_partition_sweep_cases(args)
    return build_ranked_partition_sweep_cases(args)


def build_node_contract(config: dict[str, Any], model_name: str, node_id: str) -> NodeContract:
    family_name, plan = find_model_plan(config, model_name)
    module_name = plan["node_modules"][node_id]
    module_spec = plan["modules"][module_name]
    input_edge = next((name for name, edge in plan["edges"].items() if edge["dst"] == node_id), None)
    output_edge = next((name for name, edge in plan["edges"].items() if edge["src"] == node_id), None)
    return NodeContract(
        node_id=node_id,
        model_name=model_name,
        family_name=family_name,
        module_name=module_name,
        stages=tuple(str(stage) for stage in module_spec["stages"]),
        input_edge=input_edge,
        output_edge=output_edge,
        input_payload_type=plan["edges"][input_edge]["payload"] if input_edge else None,
        output_payload_type=plan["edges"][output_edge]["payload"] if output_edge else None,
        postprocess_on_node=bool(module_spec.get("postprocess_on_node", False)),
    )


def edge_spec(config: dict[str, Any], model_name: str, edge_name: str) -> dict[str, Any]:
    _family_name, plan = find_model_plan(config, model_name)
    return plan["edges"][edge_name]


def edge_summary(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    family_name, plan = find_model_plan(config, model_name)
    return {"family": family_name, **plan}


def parse_run_name(model_name: str) -> dict[str, Any]:
    match = _RUN_RE.match(model_name)
    if match is None:
        raise ValueError(f"Invalid AsymYOLO model name: {model_name}")
    fm_freq = int(match.group("fm_freq"))
    fm_time = int(match.group("fm_time"))
    coverage = int(match.group("cpct")) / 100.0
    return {
        "variant": match.group("variant"),
        "size": match.group("size"),
        "fm_freq": fm_freq,
        "fm_time": fm_time,
        "reg_max_freq": max(1, round(coverage * fm_freq)),
        "reg_max_time": max(1, round(coverage * fm_time)),
    }


def model_path_for_name(model_name: str) -> Path:
    weights_dir = TRAINED_MODELS_DIR / model_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    return best if best.exists() else last


def load_model(model_name: str, model_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    parsed = parse_run_name(model_name)
    model = build_asym_yolo(
        variant=checkpoint.get("variant", parsed["variant"]),
        size=checkpoint.get("size", parsed["size"]),
        nc=int(checkpoint.get("nc", 2)),
        fm_freq=int(checkpoint.get("fm_freq", parsed["fm_freq"])),
        fm_time=int(checkpoint.get("fm_time", parsed["fm_time"])),
        reg_max_freq=int(checkpoint.get("reg_max_freq", parsed["reg_max_freq"])),
        reg_max_time=int(checkpoint.get("reg_max_time", parsed["reg_max_time"])),
        input_w=int(checkpoint.get("img_size", IMG_SIZE)),
        input_h=int(checkpoint.get("img_size", IMG_SIZE)),
    )
    state = checkpoint.get("ema", checkpoint.get("model"))
    if state is None:
        raise RuntimeError(f"No model state in checkpoint {model_path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).float().eval()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_network(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, int]]:
    ips = dict(DEFAULT_NODE_IPS)
    ports = dict(DEFAULT_NODE_PORTS)
    for node_id in NODES:
        ip = getattr(args, f"ip_{node_id.lower()}")
        port = getattr(args, f"port_{node_id.lower()}")
        if ip is not None:
            ips[node_id] = ip
        if port is not None:
            ports[node_id] = int(port)
    return ips, ports


def detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def value_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(value_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(value_bytes(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return 8
    return 0


def yolo_split_name(raw_split: str) -> str:
    return "val" if raw_split == "validation" else raw_split


def label_path_for_sample(yolo_root: Path, raw_split: str, sample_id: str) -> Path:
    return yolo_root / yolo_split_name(raw_split) / "labels" / f"{sample_id}.txt"


def load_yolo_label_tensor(label_path: Path) -> torch.Tensor:
    labels = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, width, height = [float(value) for value in parts[1:5]]
            labels.append(
                [
                    cls,
                    (cx - width / 2.0) * IMG_SIZE,
                    (cy - height / 2.0) * IMG_SIZE,
                    (cx + width / 2.0) * IMG_SIZE,
                    (cy + height / 2.0) * IMG_SIZE,
                ]
            )
    return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), dtype=torch.float32)
    top_left = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    bottom_right = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(2)
    area1 = (box1[:, 2:] - box1[:, :2]).clamp(min=0).prod(1)
    area2 = (box2[:, 2:] - box2[:, :2]).clamp(min=0).prod(1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-9)


def detection_correct_matrix(detections: torch.Tensor, labels: torch.Tensor, thresholds: np.ndarray) -> np.ndarray:
    correct = np.zeros((detections.shape[0], len(thresholds)), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    ious = box_iou_xyxy(detections[:, :4].float(), labels[:, 1:5].float())
    same_class = detections[:, 5:6].int() == labels[:, 0].int().view(1, -1)
    for index, threshold in enumerate(thresholds):
        det_idx, label_idx = torch.where((ious >= float(threshold)) & same_class)
        pairs = sorted(
            [(int(d), int(l), float(ious[d, l])) for d, l in zip(det_idx, label_idx)],
            key=lambda item: item[2],
            reverse=True,
        )
        used_det: set[int] = set()
        used_label: set[int] = set()
        for det, label, _iou in pairs:
            if det not in used_det and label not in used_label:
                correct[det, index] = True
                used_det.add(det)
                used_label.add(label)
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
        self.thresholds = np.linspace(0.5, 0.95, 10)
        self.correct: list[np.ndarray] = []
        self.conf: list[np.ndarray] = []
        self.pred_cls: list[np.ndarray] = []
        self.target_cls: list[np.ndarray] = []

    def add(self, detections: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        detections = detections.detach().cpu()
        labels = labels.detach().cpu()
        correct = detection_correct_matrix(detections, labels, self.thresholds)
        self.correct.append(correct)
        self.conf.append(detections[:, 4].numpy() if detections.shape[0] else np.zeros(0))
        self.pred_cls.append(detections[:, 5].numpy() if detections.shape[0] else np.zeros(0))
        self.target_cls.append(labels[:, 0].numpy() if labels.shape[0] else np.zeros(0))
        tp50 = int(correct[:, 0].sum()) if correct.size else 0
        pred_count = int(detections.shape[0])
        gt_count = int(labels.shape[0])
        return {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "tp50": tp50,
            "fp50": pred_count - tp50,
            "fn50": gt_count - tp50,
            "precision50": tp50 / pred_count if pred_count else 0.0,
            "recall50": tp50 / gt_count if gt_count else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        target_cls = np.concatenate(self.target_cls) if self.target_cls else np.zeros(0)
        correct = np.concatenate(self.correct, axis=0) if self.correct else np.zeros((0, 10), dtype=bool)
        conf = np.concatenate(self.conf) if self.conf else np.zeros(0)
        pred_cls = np.concatenate(self.pred_cls) if self.pred_cls else np.zeros(0)
        aps = np.zeros((len(CLASS_NAMES), len(self.thresholds)))
        for class_id in CLASS_NAMES:
            pred_mask = pred_cls == class_id
            n_gt = int((target_cls == class_id).sum())
            if n_gt == 0 or not pred_mask.any():
                continue
            ordered = correct[pred_mask][np.argsort(-conf[pred_mask])]
            for index in range(len(self.thresholds)):
                tp = ordered[:, index].astype(float)
                fp = 1.0 - tp
                tp_cumulative = np.cumsum(tp)
                fp_cumulative = np.cumsum(fp)
                aps[class_id, index] = compute_ap(
                    tp_cumulative / (n_gt + 1e-16),
                    tp_cumulative / (tp_cumulative + fp_cumulative + 1e-16),
                )
        total_gt = int(target_cls.shape[0])
        total_pred = int(pred_cls.shape[0])
        tp50 = int(correct[:, 0].sum()) if correct.size else 0
        return {
            "model": self.model_name,
            "image_count": len(self.target_cls),
            "target_count": total_gt,
            "prediction_count": total_pred,
            "precision": tp50 / total_pred if total_pred else 0.0,
            "recall": tp50 / total_gt if total_gt else 0.0,
            "map50": float(aps[:, 0].mean()),
            "map50_95": float(aps.mean()),
        }


def has_frequency_overlap(boxes: torch.Tensor, classes: torch.Tensor) -> bool:
    lte_boxes = boxes[classes.int() == 0]
    radar_boxes = boxes[classes.int() == 1]
    return any(max(float(lte[0]), float(radar[0])) < min(float(lte[2]), float(radar[2]))
               for lte in lte_boxes for radar in radar_boxes)


def overlap_from_detections(detections: torch.Tensor) -> bool:
    return bool(detections.shape[0] and has_frequency_overlap(detections[:, :4], detections[:, 5]))


def overlap_from_labels(labels: torch.Tensor) -> bool:
    return bool(labels.shape[0] and has_frequency_overlap(labels[:, 1:5], labels[:, 0]))


class OverlapMetricAccumulator:
    def __init__(self) -> None:
        self.tp = self.tn = self.fp = self.fn = 0

    def add(self, predicted: bool, target: bool) -> None:
        self.tp += int(predicted and target)
        self.tn += int((not predicted) and (not target))
        self.fp += int(predicted and (not target))
        self.fn += int((not predicted) and target)

    def summary(self) -> dict[str, Any]:
        total = self.tp + self.tn + self.fp + self.fn
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "overlap_accuracy": (self.tp + self.tn) / total if total else 0.0,
            "overlap_precision": precision,
            "overlap_recall": recall,
            "overlap_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }


def postprocess_raw_output(raw_output: Any, conf_threshold: float, iou_threshold: float) -> tuple[torch.Tensor, str]:
    y = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    detections = asym_non_max_suppression(
        y if y.ndim == 3 else y.unsqueeze(0),
        conf_thres=float(conf_threshold),
        iou_thres=float(iou_threshold),
        max_det=300,
    )
    result = detections[0].detach().cpu().float() if detections else torch.zeros((0, 6), dtype=torch.float32)
    return result if result.numel() else torch.zeros((0, 6), dtype=torch.float32), "local_asym_nms"


def collect_sigmf_pairs(raw_root: Path, split: str, max_samples: int | None) -> list[tuple[Path, Path]]:
    split_dir = raw_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Raw split directory not found: {split_dir}")
    pairs = []
    for meta_path in sorted(split_dir.glob("*.sigmf-meta")):
        data_path = split_dir / meta_path.name.replace(".sigmf-meta", ".sigmf-data")
        if data_path.exists():
            pairs.append((meta_path, data_path))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    return pairs


def new_runtime_state(input_tensor: dict[str, Any] | None) -> dict[str, Any]:
    return dict(input_tensor or {})


def execute_stage(
    stage: str,
    state: dict[str, Any],
    preprocessor: FastSigMFTensorPreprocessor,
    model: torch.nn.Module | None,
    device: torch.device,
    sample_paths: tuple[Path, Path] | None,
    conf_threshold: float,
    iou_threshold: float,
) -> None:
    if stage == "pre_read_iq":
        if sample_paths is None:
            raise RuntimeError("pre_read_iq requires source paths")
        state["iq_data"] = np.fromfile(sample_paths[1], dtype=np.complex64)
    elif stage == "pre_spectrogram":
        _frequencies, _times, state["sxx"] = signal.spectrogram(
            state["iq_data"],
            fs=preprocessor.sample_rate,
            nperseg=preprocessor.nfft,
            noverlap=preprocessor.noverlap,
            return_onesided=False,
            mode="magnitude",
        )
    elif stage == "pre_fftshift_log":
        state["sxx_db"] = 10 * np.log10(np.fft.fftshift(state["sxx"], axes=0) + 1e-10)
    elif stage == "pre_percentile_clip":
        low, high = preprocessor.percentile_clip
        state["vmin"] = float(np.percentile(state["sxx_db"], low))
        state["vmax"] = float(np.percentile(state["sxx_db"], high))
    elif stage == "pre_canvas_render":
        image_data = state["sxx_db"].T
        if preprocessor.image_artist is None:
            preprocessor.image_artist = preprocessor.axes.imshow(
                image_data, aspect="auto", origin="upper", cmap=preprocessor.cmap,
                vmin=state["vmin"], vmax=state["vmax"],
            )
        else:
            preprocessor.image_artist.set_data(image_data)
            preprocessor.image_artist.set_clim(state["vmin"], state["vmax"])
        preprocessor.canvas.draw()
        state["rgba"] = np.asarray(preprocessor.canvas.buffer_rgba()).copy()
    elif stage == "pre_canvas_to_rgb":
        state["rgb_image"] = np.ascontiguousarray(state["rgba"][:, :, :3].copy())
    elif stage == "pre_to_tensor":
        state["image_tensor"] = tensor_from_rgb_array(state["rgb_image"])
    elif stage.startswith("model_stage_"):
        if model is None:
            raise RuntimeError(f"{stage} requires a loaded model")
        index = int(stage.removeprefix("model_stage_"))
        state["activation"] = model.backbone[index](state.pop("image_tensor", state.get("activation")).to(device))
    elif stage == "model_sppf":
        state["activation"] = model.sppf(state["activation"])
    elif stage == "model_psa":
        state["activation"] = model.psa(state["activation"])
    elif stage == "model_head":
        state["raw_output"] = model.head(state["activation"])
    elif stage == "postprocess":
        state["detections"], state["postprocess_backend"] = postprocess_raw_output(
            state["raw_output"], conf_threshold, iou_threshold
        )
    else:
        raise RuntimeError(f"Unsupported partition stage {stage!r}")


def pack_boundary_payload(payload_type: str, state: dict[str, Any]) -> dict[str, Any]:
    if payload_type == "spectrogram_db":
        return {"sxx_db": state["sxx_db"]}
    if payload_type == "spectrogram_db_with_clip_range":
        return {"sxx_db": state["sxx_db"], "vmin": state["vmin"], "vmax": state["vmax"]}
    if payload_type == "rgba_render_buffer":
        return {"rgba": state["rgba"]}
    if payload_type == "rgb_image":
        return {"rgb_image": state["rgb_image"]}
    if payload_type == "image_tensor":
        return {"image_tensor": detach_to_cpu(state["image_tensor"])}
    if (
        payload_type.startswith("model_stage_")
        and payload_type.endswith("_output")
        or payload_type in MODEL_ACTIVATION_PAYLOAD_TYPES
    ):
        return {"activation": detach_to_cpu(state["activation"])}
    if payload_type == "decoded_prediction":
        return {"raw_output": detach_to_cpu(state["raw_output"])}
    raise RuntimeError(f"Unsupported boundary payload type {payload_type!r}")


def forward_contract_module(
    model: torch.nn.Module | None,
    contract: NodeContract,
    input_tensor: dict[str, Any] | None,
    preprocessor: FastSigMFTensorPreprocessor,
    device: torch.device,
    sample_paths: tuple[Path, Path] | None,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, float, dict[str, Any]]:
    state = new_runtime_state(move_to_device(input_tensor, device))
    t0 = time.perf_counter()
    with torch.inference_mode():
        for stage in contract.stages:
            execute_stage(stage, state, preprocessor, model, device, sample_paths, conf_threshold, iou_threshold)
    compute_sec = time.perf_counter() - t0
    outgoing = (
        pack_boundary_payload(contract.output_payload_type, state)
        if contract.output_payload_type and contract.output_payload_type != "detections"
        else None
    )
    profile = {
        "module_name": contract.module_name,
        "stages": list(contract.stages),
        "input_payload_type": contract.input_payload_type,
        "output_payload_type": contract.output_payload_type,
        "compute_sec": compute_sec,
        "transfer_tensor_bytes": value_bytes(outgoing),
        "postprocess_sec": compute_sec if "postprocess" in contract.stages else None,
        "postprocess_backend": state.get("postprocess_backend"),
        "detections_tensor_bytes": value_bytes(state.get("detections")),
    }
    return state, outgoing, compute_sec, profile


def empty_wire_stats() -> dict[str, float]:
    return {"original_bytes": 0.0, "compressed_bytes": 0.0, "compress_sec": 0.0, "decompress_sec": 0.0, "tensor_count": 0.0}


def add_wire_stats(target: dict[str, float], values: dict[str, float]) -> None:
    for key in target:
        target[key] += float(values.get(key, 0.0))


def compress_tensor_for_wire(tensor: torch.Tensor, compressor_name: str, param: Any) -> tuple[dict[str, Any], dict[str, float]]:
    cpu_tensor = detach_to_cpu(tensor)
    compressor = get_compressor(compressor_name)
    t0 = time.perf_counter()
    compressed = compressor.compress(cpu_tensor, param)
    elapsed = time.perf_counter() - t0
    return (
        {WIRE_TENSOR_MARKER: True, "compressor": compressor_name, "param": param, "data": compressed},
        {
            "original_bytes": float(value_bytes(cpu_tensor)),
            "compressed_bytes": float(compressor.get_compressed_size(compressed)),
            "compress_sec": elapsed,
            "decompress_sec": 0.0,
            "tensor_count": 1.0,
        },
    )


def compress_value_for_wire(value: Any, compressor_name: str, param: Any) -> tuple[Any, dict[str, float]]:
    stats = empty_wire_stats()
    if value is None:
        return None, stats
    if torch.is_tensor(value):
        result, child = compress_tensor_for_wire(value, compressor_name, param)
        add_wire_stats(stats, child)
        return result, stats
    if isinstance(value, dict):
        result = {}
        for key, child_value in value.items():
            result[key], child = compress_value_for_wire(child_value, compressor_name, param)
            add_wire_stats(stats, child)
        return result, stats
    if isinstance(value, list):
        result = []
        for child_value in value:
            child_result, child = compress_value_for_wire(child_value, compressor_name, param)
            result.append(child_result)
            add_wire_stats(stats, child)
        return result, stats
    if isinstance(value, tuple):
        result = []
        for child_value in value:
            child_result, child = compress_value_for_wire(child_value, compressor_name, param)
            result.append(child_result)
            add_wire_stats(stats, child)
        return tuple(result), stats
    return value, stats


def decompress_value_from_wire(value: Any, device: torch.device) -> tuple[Any, dict[str, float]]:
    stats = empty_wire_stats()
    if isinstance(value, dict) and value.get(WIRE_TENSOR_MARKER) is True:
        compressor = get_compressor(str(value["compressor"]))
        t0 = time.perf_counter()
        restored = compressor.decompress(value["data"]).to(device)
        stats.update(
            original_bytes=float(value["data"].get("original_bytes", value_bytes(restored))),
            compressed_bytes=float(compressor.get_compressed_size(value["data"])),
            decompress_sec=time.perf_counter() - t0,
            tensor_count=1.0,
        )
        return restored, stats
    if isinstance(value, dict):
        result = {}
        for key, child_value in value.items():
            result[key], child = decompress_value_from_wire(child_value, device)
            add_wire_stats(stats, child)
        return result, stats
    if isinstance(value, list):
        result = []
        for child_value in value:
            child_result, child = decompress_value_from_wire(child_value, device)
            result.append(child_result)
            add_wire_stats(stats, child)
        return result, stats
    if isinstance(value, tuple):
        result = []
        for child_value in value:
            child_result, child = decompress_value_from_wire(child_value, device)
            result.append(child_result)
            add_wire_stats(stats, child)
        return tuple(result), stats
    return value, stats


def record_wire_stats(profiles: dict[str, Any], node_id: str, edge_name: str, action: str, stats: dict[str, float]) -> None:
    if stats["tensor_count"] <= 0:
        return
    profile = profiles.setdefault(node_id, {})
    profile[f"{edge_name}_{action}_original_bytes"] = int(stats["original_bytes"])
    profile[f"{edge_name}_{action}_compressed_bytes"] = int(stats["compressed_bytes"])
    profile[f"{edge_name}_{action}_sec"] = float(stats[f"{action}_sec"])
    profile[f"{edge_name}_{action}_tensor_count"] = int(stats["tensor_count"])


def compress_payload_for_edge(
    payload: dict[str, Any],
    config: dict[str, Any],
    model_name: str,
    src: str,
    dst: str,
    compression_config: dict[str, Any],
) -> dict[str, Any]:
    edge_name = EDGE_NAMES.get((src, dst))
    if edge_name is None:
        return payload
    json_edge = edge_spec(config, model_name, f"{src}_to_{dst}")
    payload["compression_policy"] = json_edge.get("compression_policy", "forbidden")
    payload["boundary_payload_type"] = json_edge["payload"]
    if payload["compression_policy"] != "allowed" or not compression_config.get("enabled"):
        return payload
    payload["input_tensor"], stats = compress_value_for_wire(
        payload["input_tensor"],
        compression_config["compressor"],
        compression_config["edges"][edge_name]["param"],
    )
    record_wire_stats(payload.setdefault("profiles", {}), src, edge_name, "compress", stats)
    return payload


def decompress_payload_from_edge(payload: dict[str, Any], src: str, dst: str, device: torch.device) -> dict[str, Any]:
    edge_name = EDGE_NAMES.get((src, dst))
    if edge_name is None:
        return payload
    payload["input_tensor"], stats = decompress_value_from_wire(payload.get("input_tensor", {}), device)
    record_wire_stats(payload.setdefault("profiles", {}), dst, edge_name, "decompress", stats)
    return payload


def parse_rate_list(value: str) -> list[float]:
    rates = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not rates:
        raise ValueError("No rates specified")
    return rates


def compression_disabled(compressor_name: str | None) -> bool:
    return str(compressor_name or "none").strip().lower() in {"", "none", "off", "false", "0"}


def normalize_compressor_name(value: str | None) -> str:
    name = str(value or "none").strip().lower()
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
        return {"enabled": False, "label": case.label, "compressor": "none", "setting": case.setting, "param": case.param, "edges": {}}
    get_compressor(case.compressor)
    return {
        "enabled": True,
        "label": case.label,
        "compressor": case.compressor,
        "setting": case.setting,
        "param": case.param,
        "edges": {edge: {"setting": case.setting, "param": case.param} for edge in ("a_to_b", "b_to_c", "c_to_d")},
    }


def compression_case_from_spec(spec: dict[str, Any]) -> CompressionSweepCase:
    compressor_name = normalize_compressor_name(spec["compressor"])
    return CompressionSweepCase(
        label=str(spec["label"]),
        compressor=compressor_name,
        setting=str(spec.get("setting", spec.get("param"))),
        param=parse_compression_param(compressor_name, spec.get("param")),
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
    if len(set(str(value) for value in settings)) > 1:
        raise ValueError("Compression sweep requires the same setting on all links; use --compression_param.")
    setting = settings[0] if settings else default_compression_setting(compressor_name, "a_to_b")
    param = parse_compression_param(compressor_name, setting)
    label = f"{safe_name(compressor_name)}_{safe_name(setting)}"
    # Reuse the sweep label of a matching default setting so that the partition
    # rankings (keyed by these labels) can be found for a single-compressor run.
    for spec in DEFAULT_COMPRESSION_SWEEP_SPECS:
        spec_case = compression_case_from_spec(spec)
        if spec_case.compressor == compressor_name and (
            spec_case.setting == str(setting) or spec_case.param == param
        ):
            label = spec_case.label
            break
    return CompressionSweepCase(
        label=label,
        compressor=compressor_name,
        setting=str(setting),
        param=param,
        enabled=True,
    )


def build_compression_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    name = normalize_compressor_name(args.compressor)
    if name in {"sweep", "default", "all"}:
        cases = [make_uniform_compression_config(CompressionSweepCase("nocompress", "none", "none", 1.0, False))]
        cases.extend(make_uniform_compression_config(compression_case_from_spec(spec)) for spec in DEFAULT_COMPRESSION_SWEEP_SPECS)
        return cases
    return [make_uniform_compression_config(custom_compression_case(args))]


def send_with_rate(transport: Any, src: str, dst: str, msg_type: MessageType, payload: dict[str, Any], rate_mbps: float, args: argparse.Namespace) -> tuple[int, float]:
    return transport.send_dynamic(
        dst, msg_type, payload, link_idx=LINK_INDEX[(src, dst)],
        rate_bps=float(rate_mbps) * 1_000_000.0, timeout=args.request_timeout,
    )


def format_rate_value(rate_mbps: float) -> str:
    value = float(rate_mbps)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def format_rate_list(rates: list[float]) -> str:
    return "[" + ",".join(format_rate_value(value) for value in rates) + "]mbps"


def format_device_name(device_arg: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(device_arg).lower()).strip("_")
    return safe or "auto"


def format_model_name_part(model_names: list[str]) -> str:
    if len(model_names) == 1:
        return model_names[0]
    return "allmodels"


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


def partition_output_part(args: argparse.Namespace) -> str:
    if getattr(args, "partition_config_data", None) is not None:
        return "single_partition"
    return f"partition_top{int(args.partition_top_k)}"


def output_stem(args: argparse.Namespace) -> str:
    rates = parse_rate_list(args.sweep_rates_mbps)
    sample_part = "all" if args.max_samples is None else str(args.max_samples)
    return (
        f"distributedModule_FastSigmf_{format_model_name_part(args.model_names)}_{args.run_location}_"
        f"{format_device_name(args.device)}_{compression_output_part(args)}_"
        f"{partition_output_part(args)}_rates{format_rate_list(rates)}_samples{sample_part}"
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
    partition_case: dict[str, Any] | None = None,
) -> str:
    sample_part = "all" if args.max_samples is None else str(args.max_samples)
    partition_id = safe_name((partition_case or {}).get("partition_id", "single_partition"))
    return (
        f"distributedModule_FastSigmf_{model_name}_{args.run_location}_"
        f"{format_device_name(args.device)}_{compression_output_part(args, compression_config)}_"
        f"{partition_id}_rate{format_rate_value(rate_mbps)}mbps_samples{sample_part}"
    )


def case_csv_path(
    args: argparse.Namespace,
    compression_config: dict[str, Any],
    model_name: str,
    rate_mbps: float,
    partition_case: dict[str, Any] | None = None,
) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_dir / f"{case_output_stem(args, compression_config, model_name, rate_mbps, partition_case)}.csv"


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
        "partition_id",
        "partition_rank",
        "partition_cuts",
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
        "partition_id",
        "partition_rank",
        "partition_cuts",
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


def quality_worker(result_queue: queue.Queue, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.quality_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    global_metrics: dict[tuple[str, str, str, float], DetectionMetricAccumulator] = {}
    global_overlap: dict[tuple[str, str, str, float], OverlapMetricAccumulator] = {}
    cases: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    writers: dict[tuple[str, str, str, float], tuple[Any, csv.DictWriter]] = {}
    reference_cache: dict[str, tuple[torch.nn.Module, FastSigMFTensorPreprocessor]] = {}
    expected_per_case = getattr(args, "samples_per_group", None)

    def get_case(item: dict[str, Any]) -> dict[str, Any]:
        model_name = item["model_name"]
        rate = float(item["rate_mbps"])
        compression_config = item["compression_config"]
        compression_label = compression_config["label"]
        partition_id = item["partition_id"]
        key = (partition_id, compression_label, model_name, rate)
        if key not in cases:
            partition_case = {
                "partition_id": partition_id,
                "partition_rank": item["partition_rank"],
                "partition_cuts": item["partition_cuts"],
            }
            cases[key] = {
                "case_id": item["case_id"],
                "partition_id": partition_id,
                "partition_rank": item["partition_rank"],
                "partition_cuts": item["partition_cuts"],
                "partition_config_data": item["partition_config_data"],
                "compression_config": compression_config,
                "compression_label": compression_label,
                "model_name": model_name,
                "rate_mbps": rate,
                "sample_count": 0,
                "rows": [],
                "summary_written": False,
                "csv_path": case_csv_path(args, compression_config, model_name, rate, partition_case),
                "detection": DetectionMetricAccumulator(model_name),
                "overlap": OverlapMetricAccumulator(),
                "quality_errors": [],
            }
        return cases[key]

    def get_writer(case: dict[str, Any]) -> csv.DictWriter:
        key = (case["partition_id"], case["compression_label"], case["model_name"], float(case["rate_mbps"]))
        if key not in writers:
            path = case["csv_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=result_fieldnames())
            writer.writeheader()
            writers[key] = (f, writer)
        return writers[key][1]

    def get_reference(model_name: str) -> tuple[torch.nn.Module, FastSigMFTensorPreprocessor]:
        if model_name not in reference_cache:
            reference_model_path = args.model_path if args.model_path is not None else model_path_for_name(model_name)
            reference_cache[model_name] = (
                load_model(model_name, reference_model_path, torch.device("cpu")),
                FastSigMFTensorPreprocessor(),
            )
        return reference_cache[model_name]

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
        partition_summary = edge_summary(item["partition_config_data"], model_name)
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
            "partition_id": item["partition_id"],
            "partition_rank": item["partition_rank"],
            "partition_cuts": json.dumps(item["partition_cuts"], separators=(",", ":")),
            "partition_json": json.dumps(partition_summary, separators=(",", ":")),
            "A_module": partition_summary["node_modules"]["A"],
            "B_module": partition_summary["node_modules"]["B"],
            "C_module": partition_summary["node_modules"]["C"],
            "D_module": partition_summary["node_modules"]["D"],
            "A_preprocess_sec": profiles.get("A", {}).get("compute_sec"),
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

    def build_quality_summary() -> dict[str, Any]:
        def first_case_for_global_key(
            partition_id: str,
            compression_label: str,
            model_name: str,
            rate_mbps: float,
        ) -> dict[str, Any]:
            return cases[(partition_id, compression_label, model_name, float(rate_mbps))]

        return {
            "model_names": args.model_names,
            "rates_mbps": parse_rate_list(args.sweep_rates_mbps),
            "compression_cases": args.compression_cases,
            "partition_top_k": args.partition_top_k,
            "partition_case_count": len(args.sweep_cases),
            "models": {
                f"{partition_id}/{compression_label}/{model_name}@{format_rate_value(rate)}mbps": {
                    "partition": edge_summary(
                        first_case_for_global_key(partition_id, compression_label, model_name, rate)["partition_config_data"],
                        model_name,
                    ),
                    "detection": global_metrics[(partition_id, compression_label, model_name, rate)].summary(),
                    "overlap": global_overlap[(partition_id, compression_label, model_name, rate)].summary(),
                }
                for partition_id, compression_label, model_name, rate in sorted(global_metrics)
            },
            "cases": {
                f"{partition_id}/{compression_label}/{model_name}@{format_rate_value(rate)}mbps": {
                    "csv": str(case["csv_path"]),
                    "summary_csv": str(args.summary_csv),
                    "partition_id": case["partition_id"],
                    "partition_rank": case["partition_rank"],
                    "partition_cuts": case["partition_cuts"],
                    "sample_count": case["sample_count"],
                    "compression": case["compression_config"],
                    "partition": edge_summary(case["partition_config_data"], case["model_name"]),
                    "detection": case["detection"].summary(),
                    "overlap": case["overlap"].summary(),
                    "quality_error_count": len(case["quality_errors"]),
                    "quality_errors": case["quality_errors"][:20],
                }
                for (partition_id, compression_label, model_name, rate), case in sorted(cases.items())
            },
        }

    def write_quality_json() -> None:
        args.quality_json.write_text(json.dumps(build_quality_summary(), indent=2) + "\n", encoding="utf-8")

    def write_case_summary(summary_writer: csv.DictWriter, case: dict[str, Any]) -> None:
        rows = case["rows"]
        model_name = case["model_name"]
        rate_mbps = float(case["rate_mbps"])
        compression_config = case["compression_config"]
        partition_summary = edge_summary(case["partition_config_data"], model_name)
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
                "partition_id": case["partition_id"],
                "partition_rank": case["partition_rank"],
                "partition_cuts": json.dumps(case["partition_cuts"], separators=(",", ":")),
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
    write_quality_json()
    try:
        while True:
            item = result_queue.get()
            try:
                if item is None:
                    break

                result_payload = item["result_envelope"]["payload"]
                detections = result_payload["detections"]
                profiles = result_payload.get("profiles", {})
                if args.verify_local_reference:
                    reference_model, reference_preprocessor = get_reference(item["model_name"])
                    reference_tensor = reference_preprocessor.raw_to_tensor(item["meta_path"], item["data_path"])
                    with torch.inference_mode():
                        reference_raw = reference_model(reference_tensor)
                    reference_detections, _ = postprocess_raw_output(
                        reference_raw,
                        args.conf_threshold,
                        args.iou_threshold,
                    )
                    torch.testing.assert_close(detections, reference_detections, rtol=0.0, atol=0.0)

                case = get_case(item)
                writer = get_writer(case)
                model_name = item["model_name"]
                global_key = (item["partition_id"], item["compression_config"]["label"], model_name, float(item["rate_mbps"]))
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

                    pred_overlap = overlap_from_detections(detections)
                    gt_overlap = overlap_from_labels(labels)
                    sample_overlap.add(pred_overlap, gt_overlap)
                    global_overlap[global_key].add(pred_overlap, gt_overlap)
                    case["overlap"].add(pred_overlap, gt_overlap)
                except Exception as exc:
                    quality_error = repr(exc)
                    case["quality_errors"].append(f"{item['sample_id']}: {quality_error}")
                    print(f"[A][quality] failed sample={item['sample_id']}: {quality_error}", flush=True)

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
                        item["partition_id"],
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
                    write_quality_json()
            finally:
                result_queue.task_done()
    finally:
        for f, _writer in writers.values():
            f.close()
        for _key, case in sorted(cases.items()):
            if case["sample_count"] > 0 and not case["summary_written"]:
                write_case_summary(summary_writer, case)
        summary_f.flush()
        summary_f.close()
        write_quality_json()

    for (_partition_id, _compression_label, _model_name, _rate), case in sorted(cases.items()):
        print(f"[A][quality] csv written to {case['csv_path']}", flush=True)
    print(f"[A][quality] summary csv written to {args.summary_csv}", flush=True)
    print(f"[A][quality] summary written to {args.quality_json}", flush=True)


def run_node_a(args: argparse.Namespace) -> None:
    if args.max_inflight <= 0:
        raise ValueError("--max_inflight must be positive")

    ips, ports = resolve_network(args)
    device = resolve_device(args.device)
    pairs = collect_sigmf_pairs(args.raw_root, args.split, args.max_samples)
    args.samples_per_group = len(pairs)
    transport = create_transport("A", ips, ports, request_timeout=args.request_timeout, rate_mode="dynamic", tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes)
    transport.start()
    print(
        f"[A] ready, listening on {ips['A']}:{ports['A']}, device={device}, "
        f"models={','.join(args.model_names)}, samples={len(pairs)}, sweep_cases={len(args.sweep_cases)}",
        flush=True,
    )
    transport.wait_for_peers(WORKER_NODES, timeout=args.ready_timeout)
    print(f"[A] peers ready: {','.join(WORKER_NODES)}", flush=True)

    task_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    send_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    result_queue: queue.Queue = queue.Queue(maxsize=args.quality_queue_size)
    error_queue: queue.Queue = queue.Queue()
    inflight_slots = threading.Semaphore(args.max_inflight)
    inflight_lock = threading.Lock()
    inflight: dict[str, dict[str, Any]] = {}
    counters = {"sent": 0, "received": 0}
    send_done_event = threading.Event()

    def prepare_case(sweep_case: dict[str, Any]) -> dict[str, Any]:
        compression_config = sweep_case["compression_config"]
        model_name = sweep_case["model_name"]
        rate_mbps = float(sweep_case["rate_mbps"])
        partition_id = sweep_case["partition_id"]
        partition_config_data = sweep_case["partition_config_data"]
        case_id = f"{compression_config['label']}_{model_name}@{format_rate_value(rate_mbps)}mbps_{partition_id}"
        t0 = time.perf_counter()
        transport.reset_dynamic_channels()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        payload = {
            "cmd": "prepare_case",
            "case_id": case_id,
            "model_name": model_name,
            "rate_mbps": rate_mbps,
            "partition_id": partition_id,
            "partition_rank": sweep_case["partition_rank"],
            "partition_cuts": sweep_case["partition_cuts"],
            "partition_config_data": partition_config_data,
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

        contract = build_node_contract(partition_config_data, model_name, "A")
        preprocessor = FastSigMFTensorPreprocessor()
        model_path = args.model_path if args.model_path is not None else model_path_for_name(model_name)
        model = load_model(model_name, model_path, device) if any(stage.startswith("model_") for stage in contract.stages) else None
        case_started_at = time.perf_counter()
        prepare_sec = case_started_at - t0
        print(
            f"[A] case_ready case={case_id} prepare={prepare_sec:.4f}s "
            f"partition={partition_id} workers={','.join(sorted(ready_nodes))}",
            flush=True,
        )
        return {
            "case_id": case_id,
            "partition_id": partition_id,
            "partition_rank": sweep_case["partition_rank"],
            "partition_cuts": sweep_case["partition_cuts"],
            "partition_config_data": partition_config_data,
            "compression_config": compression_config,
            "compression_label": compression_config["label"],
            "model_name": model_name,
            "rate_mbps": rate_mbps,
            "case_started_at": case_started_at,
            "case_prepare_sec": prepare_sec,
            "contract": contract,
            "preprocessor": preprocessor,
            "model": model,
        }

    def producer_worker() -> None:
        def wait_for_group_drain(partition_id: str, compression_label: str, model_name: str, rate_mbps: float) -> None:
            while True:
                if not error_queue.empty():
                    return
                with inflight_lock:
                    pending = any(
                        meta.get("partition_id") == partition_id
                        and meta.get("compression_label") == compression_label
                        and meta.get("model_name") == model_name
                        and float(meta.get("rate_mbps")) == float(rate_mbps)
                        for meta in inflight.values()
                    )
                if not pending:
                    return
                time.sleep(0.1)

        try:
            for sweep_case in args.sweep_cases:
                compression_config = sweep_case["compression_config"]
                model_name = sweep_case["model_name"]
                rate_mbps = float(sweep_case["rate_mbps"])
                partition_id = sweep_case["partition_id"]
                case_meta = prepare_case(sweep_case)
                for meta_path, data_path in pairs:
                    if not error_queue.empty():
                        return
                    inflight_slots.acquire()
                    task_id = uuid.uuid4().hex
                    sample_id = meta_path.name.replace(".sigmf-meta", "")
                    started_at = time.perf_counter()
                    with inflight_lock:
                        inflight[task_id] = {
                            "case_id": case_meta["case_id"],
                            "partition_id": partition_id,
                            "partition_rank": case_meta["partition_rank"],
                            "partition_cuts": case_meta["partition_cuts"],
                            "partition_config_data": case_meta["partition_config_data"],
                            "compression_config": compression_config,
                            "compression_label": compression_config["label"],
                            "sample_id": sample_id,
                            "model_name": model_name,
                            "started_at": started_at,
                            "case_started_at": case_meta["case_started_at"],
                            "case_prepare_sec": case_meta["case_prepare_sec"],
                            "rate_mbps": rate_mbps,
                            "send_done": None,
                            "a_to_b_payload_bytes": None,
                            "a_to_b_send_sec": None,
                            "case_prepare_sec": case_meta["case_prepare_sec"],
                            "meta_path": meta_path,
                            "data_path": data_path,
                        }
                    try:
                        state, outgoing, _compute, profile = forward_contract_module(
                            case_meta["model"],
                            case_meta["contract"],
                            None,
                            case_meta["preprocessor"],
                            device,
                            (meta_path, data_path),
                            args.conf_threshold,
                            args.iou_threshold,
                        )
                        task_queue.put(
                            {
                                "task_id": task_id,
                                "case_id": case_meta["case_id"],
                                "partition_id": partition_id,
                                "partition_rank": case_meta["partition_rank"],
                                "partition_cuts": case_meta["partition_cuts"],
                                "partition_config_data": case_meta["partition_config_data"],
                                "compression_config": compression_config,
                                "sample_id": sample_id,
                                "model_name": model_name,
                                "rate_mbps": rate_mbps,
                                "input_tensor": detach_to_cpu(outgoing),
                                "profiles": {"A": profile},
                            }
                        )
                    except Exception:
                        with inflight_lock:
                            if inflight.pop(task_id, None) is not None:
                                inflight_slots.release()
                        raise
                wait_for_group_drain(partition_id, compression_config["label"], model_name, rate_mbps)
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
                        meta["send_done"] = send_done
                    payload = {
                        "cmd": "infer",
                        "case_id": task["case_id"],
                        "task_id": task["task_id"],
                        "sample_id": task["sample_id"],
                        "model_name": task["model_name"],
                        "rate_mbps": task["rate_mbps"],
                        "partition_id": task["partition_id"],
                        "partition_rank": task["partition_rank"],
                        "partition_cuts": task["partition_cuts"],
                        "compression_config": task["compression_config"],
                        "input_tensor": task["input_tensor"],
                        "tensor_layers": {},
                        "profiles": task["profiles"],
                    }
                    payload = compress_payload_for_edge(
                        payload,
                        task["partition_config_data"],
                        task["model_name"],
                        "A",
                        "B",
                        task["compression_config"],
                    )
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
                        f"task_id={task_id}",
                        flush=True,
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
                            raise TimeoutError("Timed out waiting for distributed AsymYOLO results")
                    continue

                result = result_envelope["payload"]
                task_id = result.get("task_id")
                with inflight_lock:
                    meta = inflight.get(task_id)
                if meta is None:
                    print(f"[A] ignored unknown result task_id={task_id}", flush=True)
                    continue

                result.setdefault("profiles", {}).setdefault("D", {})
                result["profiles"]["D"]["send_payload_bytes"] = int(result_envelope.get("payload_bytes", 0))
                result["profiles"]["D"]["send_sec"] = float(result_envelope.get("transfer_elapsed_sec", 0.0))
                meta["send_done"].wait()
                now = time.perf_counter()
                result_queue.put(
                    {
                        "case_id": meta["case_id"],
                        "partition_id": meta["partition_id"],
                        "partition_rank": meta["partition_rank"],
                        "partition_cuts": meta["partition_cuts"],
                        "partition_config_data": meta["partition_config_data"],
                        "compression_config": meta["compression_config"],
                        "task_id": task_id,
                        "sample_id": meta["sample_id"],
                        "model_name": meta["model_name"],
                        "rate_mbps": meta["rate_mbps"],
                        "a_to_b_payload_bytes": meta["a_to_b_payload_bytes"],
                        "a_to_b_send_sec": meta["a_to_b_send_sec"],
                        "case_prepare_sec": meta["case_prepare_sec"],
                        "started_at_sec": meta["started_at"] - meta["case_started_at"],
                        "end_to_end_sec": now - meta["started_at"],
                        "completed_at_sec": now - meta["case_started_at"],
                        "meta_path": meta["meta_path"],
                        "data_path": meta["data_path"],
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

    quality_thread = threading.Thread(target=quality_worker, args=(result_queue, args), daemon=False)
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
        shutdown_rate = args.rates_mbps[0] if args.rates_mbps else 1.0
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
        if not error_queue.empty():
            raise error_queue.get()


def run_worker(node_id: str, args: argparse.Namespace) -> None:
    if args.max_inflight <= 0:
        raise ValueError("--max_inflight must be positive")

    ips, ports = resolve_network(args)
    device = resolve_device(args.device)
    transport = create_transport(node_id, ips, ports, request_timeout=args.request_timeout, rate_mode="dynamic", tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes)
    transport.start()
    print(f"[{node_id}] ready, listening on {ips[node_id]}:{ports[node_id]}, device={device}, models={','.join(args.model_names)}", flush=True)

    inference_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    send_queue: queue.Queue = queue.Queue(maxsize=args.max_inflight)
    error_queue: queue.Queue = queue.Queue()
    current: dict[str, Any] = {}
    runtime_lock = threading.Lock()

    def prepare_runtime_for_case(
        case_id: str,
        model_name: str,
        rate_mbps: float,
        partition_id: str,
        partition_rank: int,
        partition_cuts: list[Any],
        partition_config_data: dict[str, Any],
        compression_config: dict[str, Any],
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        with runtime_lock:
            transport.reset_dynamic_channels()
            current.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            contract = build_node_contract(partition_config_data, model_name, node_id)
            need_model = any(stage.startswith("model_") for stage in contract.stages)
            model_path = args.model_path if args.model_path is not None else model_path_for_name(model_name)
            current.update(
                {
                    "case_id": case_id,
                    "partition_id": partition_id,
                    "partition_rank": partition_rank,
                    "partition_cuts": partition_cuts,
                    "partition_config_data": partition_config_data,
                    "model_name": model_name,
                    "rate_mbps": float(rate_mbps),
                    "compression_config": compression_config,
                    "contract": contract,
                    "preprocessor": FastSigMFTensorPreprocessor(),
                    "model": load_model(model_name, model_path, device) if need_model else None,
                }
            )
        load_sec = time.perf_counter() - t0
        print(
            f"[{node_id}] case_ready case={case_id}, model={model_name}, "
            f"rate={float(rate_mbps):g}Mbps, partition={partition_id}, load={load_sec:.4f}s, module={contract.module_name}, "
            f"stages={','.join(contract.stages)}",
            flush=True,
        )
        return {
            "cmd": "case_ready",
            "case_id": case_id,
            "partition_id": partition_id,
            "partition_rank": partition_rank,
            "partition_cuts": partition_cuts,
            "node": node_id,
            "model_name": model_name,
            "rate_mbps": float(rate_mbps),
            "compression_config": compression_config,
            "module_name": contract.module_name,
            "stages": list(contract.stages),
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
                        partition_id=payload.get("partition_id", "single_partition"),
                        partition_rank=int(payload.get("partition_rank", 1)),
                        partition_cuts=list(payload.get("partition_cuts", [])),
                        partition_config_data=payload.get("partition_config_data", args.partition_config_data),
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
                    partition_id = payload.get("partition_id", current.get("partition_id"))
                    profiles = payload.setdefault("profiles", {})
                    src = envelope.get("src")
                    if src in NODES:
                        profiles.setdefault(src, {})
                        profiles[src]["send_payload_bytes"] = int(envelope.get("payload_bytes", 0))
                        profiles[src]["send_sec"] = float(envelope.get("transfer_elapsed_sec", 0.0))
                    decompress_payload_from_edge(payload, src, node_id, device)

                    with runtime_lock:
                        if current.get("case_id") != case_id or current.get("model_name") != model_name:
                            raise RuntimeError(
                                f"[{node_id}] received task before matching prepare_case: "
                                f"case={case_id}, model={model_name}, "
                                f"current_case={current.get('case_id')}, current_model={current.get('model_name')}"
                            )
                        state, outgoing, compute_sec, profile = forward_contract_module(
                            current["model"],
                            current["contract"],
                            payload.get("input_tensor"),
                            current["preprocessor"],
                            device,
                            None,
                            args.conf_threshold,
                            args.iou_threshold,
                        )
                        rate_mbps = float(current["rate_mbps"])
                        compression_config = current["compression_config"]
                        partition_config_data = current["partition_config_data"]
                    profiles[node_id] = {**profiles.get(node_id, {}), **profile}
                    dst = NEXT_HOP[node_id]
                    msg_type = NEXT_MESSAGE[node_id]
                    if node_id == "D":
                        out_payload = {
                            "cmd": "result",
                            "case_id": case_id,
                            "task_id": task_id,
                            "sample_id": payload["sample_id"],
                            "model_name": model_name,
                            "rate_mbps": payload["rate_mbps"],
                            "partition_id": partition_id,
                            "partition_rank": payload.get("partition_rank", current.get("partition_rank")),
                            "partition_cuts": payload.get("partition_cuts", current.get("partition_cuts")),
                            "compression_config": compression_config,
                            "detections": detach_to_cpu(state["detections"]),
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
                            "partition_id": partition_id,
                            "partition_rank": payload.get("partition_rank", current.get("partition_rank")),
                            "partition_cuts": payload.get("partition_cuts", current.get("partition_cuts")),
                            "compression_config": compression_config,
                            "input_tensor": outgoing,
                            "tensor_layers": {},
                            "profiles": profiles,
                        }
                        out_payload = compress_payload_for_edge(
                            out_payload,
                            partition_config_data,
                            model_name,
                            node_id,
                            dst,
                            compression_config,
                        )
                    send_queue.put(
                        {
                            "task_id": task_id,
                            "dst": dst,
                            "msg_type": msg_type,
                            "payload": out_payload,
                            "rate_mbps": rate_mbps,
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
                        f"send->{item['dst']}={send_sec:.4f}s rate={item['rate_mbps']}Mbps",
                        flush=True,
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
    global TRAINED_MODELS_DIR
    parser = argparse.ArgumentParser(description="Distributed FastSigMF AsymYOLO stage-sweep inference over HTTP")
    parser.add_argument("--node", choices=NODES, required=True)
    for node_id in NODES:
        parser.add_argument(f"--ip_{node_id.lower()}", default=None)
        parser.add_argument(f"--port_{node_id.lower()}", type=int, default=None)
    parser.add_argument("--model_name", default=None, help="Run one model. If omitted, sweep --sweep_models.")
    parser.add_argument("--sweep_models", default="all", help="Comma-separated models or 'all'.")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--models_dir", type=Path, default=TRAINED_MODELS_DIR,
                        help="Directory containing sig_yolo_*/weights/best.pt on every node.")
    parser.add_argument(
        "--partition_json",
        type=Path,
        default=None,
        help=f"Optional single-partition JSON. If omitted, sweep top ranked partitions from configs/ rankings.",
    )
    parser.add_argument("--partition_rankings_nocompress_json", type=Path, default=DEFAULT_NOCOMPRESS_PARTITION_RANKINGS_JSON)
    parser.add_argument("--partition_rankings_compressed_json", type=Path, default=DEFAULT_COMPRESSED_PARTITION_RANKINGS_JSON)
    parser.add_argument("--partition_top_k", type=int, default=DEFAULT_PARTITION_TOP_K)
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
    parser.add_argument("--verify_local_reference", action="store_true")
    args = parser.parse_args()
    if args.partition_top_k <= 0:
        raise ValueError("--partition_top_k must be positive")
    if args.partition_json is not None:
        args.partition_config_data = load_partition_config(args.partition_json)
        supported = supported_model_names(args.partition_config_data)
        args.partition_rankings_nocompress_data = {}
        args.partition_rankings_compressed_data = {}
    else:
        args.partition_config_data = None
        args.partition_rankings_nocompress_data = load_partition_config(args.partition_rankings_nocompress_json)
        args.partition_rankings_compressed_data = load_partition_config(args.partition_rankings_compressed_json)
        supported = ranking_model_names(args.partition_rankings_nocompress_data, args.partition_rankings_compressed_data)
    TRAINED_MODELS_DIR = args.models_dir
    args.model_names = resolve_model_names(args.model_name, args.sweep_models, supported)
    args.rates_mbps = parse_rate_list(args.sweep_rates_mbps)
    args.compression_cases = build_compression_cases(args)
    args.sweep_cases = build_partition_sweep_cases(args)
    for case in args.sweep_cases:
        validate_partition_boundaries(case["partition_config_data"], source=f"partition {case['partition_id']}")
    if args.model_path is not None and len(args.model_names) != 1:
        raise ValueError("--model_path can only be used with a single --model_name")
    if args.output_csv is not None and len(args.sweep_cases) != 1:
        raise ValueError("--output_csv can only be used with one fully resolved sweep case")
    args.model_name = args.model_names[0] if len(args.model_names) == 1 else format_model_name_part(args.model_names)
    if args.model_path is None and len(args.model_names) == 1:
        args.model_path = model_path_for_name(args.model_names[0])
    resolve_output_paths(args)
    if args.verify_local_reference and any(case["enabled"] for case in args.compression_cases):
        raise ValueError("--verify_local_reference requires --compressor none")
    return args


def main() -> None:
    args = parse_args()
    if args.node == "A":
        run_node_a(args)
    else:
        run_worker(args.node, args)


if __name__ == "__main__":
    main()
