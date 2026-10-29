"""
Sequential local stage cost test for CBRS detection models on raw SigMF samples.

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
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

from cbrs.sigmf_fast_tensor_preprocessor import FastSigMFTensorPreprocessor, tensor_from_rgb_array


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_baseline_results_by_stage_cost"
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
# Sample limit: 100 = first 100 valid samples; 0 or negative = all samples (default).
# --max_samples overrides this value; the inference batch size is always 1.
MAX_SAMPLES = 0
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

PREPROCESS_STAGES = [
    "pre_read_iq",
    "pre_spectrogram",
    "pre_fftshift_log",
    "pre_percentile_clip",
    "pre_canvas_render",
    "pre_canvas_to_rgb",
    "pre_to_tensor",
]
MODEL_STAGES = [
    "model_stem",
    "model_p2",
    "model_p3",
    "model_p4",
    "model_p5",
    "model_neck",
    "model_encoder",
    "model_decoder",
    "model_head",
]
POSTPROCESS_STAGES = [
    "post_raw_decode",
    "post_load_label",
    "post_detection_metric",
    "post_overlap_metric",
]
PROFILE_NUMERIC_FIELDS = (
    [f"{stage}_{suffix}" for stage in PREPROCESS_STAGES for suffix in ("sec", "bytes")]
    + [f"{stage}_{suffix}" for stage in MODEL_STAGES for suffix in ("sec", "bytes")]
    + [
        "model_backbone_to_neck_bytes",
        "model_backbone_to_encoder_bytes",
        "model_encoder_to_decoder_bytes",
    ]
    + [f"{stage}_{suffix}" for stage in POSTPROCESS_STAGES for suffix in ("sec", "bytes")]
)
PROFILE_STRING_FIELDS = [
    "model_backbone_to_neck_tensors",
    "model_backbone_to_encoder_tensors",
    "model_encoder_to_decoder_tensors",
]


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


def empty_profile() -> dict[str, Any]:
    profile = {key: 0.0 for key in PROFILE_NUMERIC_FIELDS}
    for key in PROFILE_STRING_FIELDS:
        profile[key] = ""
    return profile


def empty_stage_profile(stages: list[str]) -> dict[str, Any]:
    return {f"{stage}_{suffix}": 0.0 for stage in stages for suffix in ("sec", "bytes")}


def empty_model_profile() -> dict[str, Any]:
    profile = empty_stage_profile(MODEL_STAGES)
    profile.update(
        {
            "model_backbone_to_neck_bytes": 0.0,
            "model_backbone_to_encoder_bytes": 0.0,
            "model_encoder_to_decoder_bytes": 0.0,
            "model_backbone_to_neck_tensors": "",
            "model_backbone_to_encoder_tensors": "",
            "model_encoder_to_decoder_tensors": "",
        }
    )
    return profile


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


def normalize_max_samples(max_samples: int | None) -> int | None:
    """None, 0 and negative values all mean no sample limit."""
    return max_samples if max_samples is not None and max_samples > 0 else None


def collect_sigmf_pairs(raw_root: Path, split: str, max_samples: int | None) -> list[tuple[Path, Path]]:
    max_samples = normalize_max_samples(max_samples)
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


def tensor_bytes(value: Any) -> int:
    seen: set[int] = set()

    def walk(item: Any) -> int:
        if torch.is_tensor(item):
            item_id = id(item)
            if item_id in seen:
                return 0
            seen.add(item_id)
            return int(item.numel() * item.element_size())
        if isinstance(item, np.ndarray):
            item_id = id(item)
            if item_id in seen:
                return 0
            seen.add(item_id)
            return int(item.nbytes)
        if isinstance(item, bytes):
            return len(item)
        if isinstance(item, dict):
            return sum(walk(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(walk(child) for child in item)
        return 0

    return walk(value)


def profiled_preprocess_sigmf_to_tensor(
    meta_path: Path,
    data_path: Path,
    preprocessor: FastSigMFTensorPreprocessor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _ = meta_path
    profile = empty_stage_profile(PREPROCESS_STAGES)

    t0 = time.perf_counter()
    iq_data = np.fromfile(Path(data_path), dtype=np.complex64)
    profile["pre_read_iq_sec"] = time.perf_counter() - t0
    profile["pre_read_iq_bytes"] = int(iq_data.nbytes)

    t0 = time.perf_counter()
    _frequencies, _times, sxx = signal.spectrogram(
        iq_data,
        fs=preprocessor.sample_rate,
        nperseg=preprocessor.nfft,
        noverlap=preprocessor.noverlap,
        return_onesided=False,
        mode="magnitude",
    )
    profile["pre_spectrogram_sec"] = time.perf_counter() - t0
    profile["pre_spectrogram_bytes"] = int(sxx.nbytes)

    t0 = time.perf_counter()
    sxx = np.fft.fftshift(sxx, axes=0)
    sxx_db = 10 * np.log10(sxx + 1e-10)
    profile["pre_fftshift_log_sec"] = time.perf_counter() - t0
    profile["pre_fftshift_log_bytes"] = int(sxx_db.nbytes)

    t0 = time.perf_counter()
    low, high = preprocessor.percentile_clip
    vmin = np.percentile(sxx_db, low)
    vmax = np.percentile(sxx_db, high)
    profile["pre_percentile_clip_sec"] = time.perf_counter() - t0
    profile["pre_percentile_clip_bytes"] = 16

    t0 = time.perf_counter()
    image_data = sxx_db.T
    if preprocessor.image_artist is None:
        preprocessor.image_artist = preprocessor.axes.imshow(
            image_data,
            aspect="auto",
            origin="upper",
            cmap=preprocessor.cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        preprocessor.image_artist.set_data(image_data)
        preprocessor.image_artist.set_clim(vmin, vmax)
    preprocessor.canvas.draw()
    profile["pre_canvas_render_sec"] = time.perf_counter() - t0
    profile["pre_canvas_render_bytes"] = int(preprocessor.image_size[0] * preprocessor.image_size[1] * 4)

    t0 = time.perf_counter()
    rgba = np.asarray(preprocessor.canvas.buffer_rgba())
    image = np.ascontiguousarray(rgba[:, :, :3].copy())
    profile["pre_canvas_to_rgb_sec"] = time.perf_counter() - t0
    profile["pre_canvas_to_rgb_bytes"] = int(image.nbytes)

    t0 = time.perf_counter()
    tensor = tensor_from_rgb_array(image)
    profile["pre_to_tensor_sec"] = time.perf_counter() - t0
    profile["pre_to_tensor_bytes"] = tensor_bytes(tensor)
    return tensor, profile


def profile_stage_total(profile: dict[str, Any], stages: list[str]) -> float:
    return float(sum(float(profile.get(f"{stage}_sec", 0.0) or 0.0) for stage in stages))


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


def get_layers(model) -> list[Any]:
    layers = getattr(model.model, "model", None)
    if layers is None:
        raise RuntimeError("Could not find Ultralytics model.model layers")
    return list(layers)


def resolve_layer_ref(outputs: list[Any], current_index: int, ref: int, current_output: Any) -> Any:
    if ref == -1:
        return current_output
    index = current_index + ref if ref < 0 else ref
    if index < 0 or index >= len(outputs) or outputs[index] is None:
        raise RuntimeError(f"Missing layer output for ref={ref} at layer={current_index}")
    return outputs[index]


def layer_input(module, outputs: list[Any], current_index: int, current_output: Any) -> Any:
    f = getattr(module, "f", -1)
    if isinstance(f, int):
        return resolve_layer_ref(outputs, current_index, f, current_output)
    return [resolve_layer_ref(outputs, current_index, int(ref), current_output) for ref in f]


def forward_layer_span(
    layers: list[Any],
    outputs: list[Any],
    current_output: Any,
    start: int,
    end: int,
) -> Any:
    for index in range(start, end):
        module = layers[index]
        current_output = module(layer_input(module, outputs, index, current_output))
        outputs[index] = current_output
    return current_output


def run_profiled_stage(
    profile: dict[str, Any],
    stage_name: str,
    layers: list[Any],
    outputs: list[Any],
    current_output: Any,
    start: int,
    end: int,
) -> Any:
    if torch.cuda.is_available():
        first_tensor = first_tensor_in(current_output)
        if first_tensor is not None and first_tensor.is_cuda:
            torch.cuda.synchronize(first_tensor.device)
    t0 = time.perf_counter()
    current_output = forward_layer_span(layers, outputs, current_output, start, end)
    if current_output is not None and torch.cuda.is_available():
        first_tensor = first_tensor_in(current_output)
        if first_tensor is not None and first_tensor.is_cuda:
            torch.cuda.synchronize(first_tensor.device)
    profile[f"{stage_name}_sec"] = time.perf_counter() - t0
    profile[f"{stage_name}_bytes"] = tensor_bytes(current_output)
    return current_output


def first_tensor_in(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for child in value.values():
            found = first_tensor_in(child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = first_tensor_in(child)
            if found is not None:
                return found
    return None


def yolo_profile_plan(model_name: str, layer_count: int) -> dict[str, Any]:
    if "yolo8" in model_name and layer_count == 23:
        return {
            "ranges": {
                "model_stem": (0, 1),
                "model_p2": (1, 3),
                "model_p3": (3, 5),
                "model_p4": (5, 7),
                "model_p5": (7, 10),
                "model_neck": (10, 22),
                "model_head": (22, 23),
            },
            "backbone_to_neck": [("P3", 4), ("P4", 6), ("P5", 9)],
            "neck_to_head": [("N3", 15), ("N4", 18), ("N5", 21)],
        }
    if ("yolo11" in model_name or "yolo26" in model_name) and layer_count == 24:
        return {
            "ranges": {
                "model_stem": (0, 1),
                "model_p2": (1, 3),
                "model_p3": (3, 5),
                "model_p4": (5, 7),
                "model_p5": (7, 11),
                "model_neck": (11, 23),
                "model_head": (23, 24),
            },
            "backbone_to_neck": [("P3", 4), ("P4", 6), ("P5", 10)],
            "neck_to_head": [("N3", 16), ("N4", 19), ("N5", 22)],
        }
    raise RuntimeError(f"Unsupported YOLO profile layout: model={model_name}, layers={layer_count}")


def rtdetr_profile_plan(layer_count: int) -> dict[str, Any]:
    if layer_count != 29:
        raise RuntimeError(f"Unsupported RT-DETR profile layout: layers={layer_count}")
    return {
        "ranges": {
            "model_stem": (0, 1),
            "model_p2": (1, 4),
            "model_p3": (4, 8),
            "model_p4": (8, 10),
            "model_p5": (10, 13),
            "model_encoder": (13, 28),
            "model_decoder": (28, 29),
        },
        "backbone_to_encoder": [("P3", 3), ("P4", 7), ("P5", 12)],
        "encoder_to_decoder": [("E3", 21), ("E4", 24), ("E5", 27)],
    }


def boundary_payload(outputs: list[Any], refs: list[tuple[str, int]]) -> tuple[int, str]:
    tensors = []
    metadata = []
    for name, index in refs:
        value = outputs[index]
        if value is None:
            raise RuntimeError(f"Missing boundary tensor {name} at layer {index}")
        bytes_count = tensor_bytes(value)
        tensors.append(value)
        metadata.append({"name": name, "layer": index, "bytes": bytes_count})
    return tensor_bytes(tensors), json.dumps(metadata)


def profiled_model_forward(
    model_name: str,
    model,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    layers = get_layers(model)
    outputs: list[Any] = [None] * len(layers)
    profile = empty_model_profile()
    current_output: Any = x.to(device)
    schema: dict[str, Any] = {
        "model_name": model_name,
        "layer_count": len(layers),
        "stage_ranges": {},
        "boundary_tensors": {},
    }

    with torch.inference_mode():
        if is_rtdetr_model(model_name):
            plan = rtdetr_profile_plan(len(layers))
            for stage_name, (start, end) in plan["ranges"].items():
                current_output = run_profiled_stage(profile, stage_name, layers, outputs, current_output, start, end)
                schema["stage_ranges"][stage_name] = [start, end]
                if stage_name == "model_encoder":
                    encoder_bytes, encoder_meta = boundary_payload(outputs, plan["encoder_to_decoder"])
                    profile["model_encoder_bytes"] = encoder_bytes
                    profile["model_encoder_to_decoder_bytes"] = encoder_bytes
                    profile["model_encoder_to_decoder_tensors"] = encoder_meta
                    schema["boundary_tensors"]["encoder_to_decoder"] = json.loads(encoder_meta)

            backbone_bytes, backbone_meta = boundary_payload(outputs, plan["backbone_to_encoder"])
            profile["model_backbone_to_encoder_bytes"] = backbone_bytes
            profile["model_backbone_to_encoder_tensors"] = backbone_meta
            schema["boundary_tensors"]["backbone_to_encoder"] = json.loads(backbone_meta)
            profile["model_head_sec"] = 0.0
            profile["model_head_bytes"] = tensor_bytes(current_output)
            return current_output, profile, schema

        plan = yolo_profile_plan(model_name, len(layers))
        for stage_name, (start, end) in plan["ranges"].items():
            current_output = run_profiled_stage(profile, stage_name, layers, outputs, current_output, start, end)
            schema["stage_ranges"][stage_name] = [start, end]
            if stage_name == "model_p5":
                boundary_bytes, boundary_meta = boundary_payload(outputs, plan["backbone_to_neck"])
                profile["model_backbone_to_neck_bytes"] = boundary_bytes
                profile["model_backbone_to_neck_tensors"] = boundary_meta
                schema["boundary_tensors"]["backbone_to_neck"] = json.loads(boundary_meta)
            if stage_name == "model_neck":
                head_bytes, head_meta = boundary_payload(outputs, plan["neck_to_head"])
                profile["model_neck_bytes"] = head_bytes
                schema["boundary_tensors"]["neck_to_head"] = json.loads(head_meta)

    return current_output, profile, schema


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


def filename_token(value: Any) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in str(value))


def output_parameter_tag(args: argparse.Namespace, device_name: str) -> str:
    # samplesN = requested limit; samplesall = no limit. The processed count is stored in the results.
    sample_limit = normalize_max_samples(args.max_samples)
    samples = "all" if sample_limit is None else str(sample_limit)
    conf = filename_token(str(args.conf_threshold).replace(".", "p"))
    iou = filename_token(str(args.iou_threshold).replace(".", "p"))
    location = "" if args.run_location == "local" else f"_at-{filename_token(args.run_location)}"
    return (
        f"{filename_token(device_name)}_{args.split}_bs1_img{IMG_SIZE}_"
        f"samples{samples}_conf{conf}_iou{iou}{location}"
    )


def output_stem(args: argparse.Namespace) -> str:
    family = "rtdetr" if is_rtdetr_model(args.model_name) else "yolo"
    return (
        f"local_original_{family}_{args.model_name}_FastSigmf_stagecost_"
        f"{output_parameter_tag(args, args.device)}"
    )


def sweep_output_stem(args: argparse.Namespace) -> str:
    models = sorted({resolve_model_name(name) for name in parse_csv_list(args.sweep_models)})
    devices = "-".join(parse_csv_list(args.sweep_devices))
    families = "_".join(sorted({"rtdetr" if is_rtdetr_model(name) else "yolo" for name in models}))
    if len(models) == 1:
        model_tag = models[0]
    elif set(models) == set(DEFAULT_SWEEP_MODELS):
        model_tag = "allmodels"
    else:
        # Distinguish sweeps with the same model count but different model sets without very long file names.
        digest = hashlib.sha256(",".join(models).encode("utf-8")).hexdigest()[:8]
        model_tag = f"models{len(models)}-{digest}"
    return (
        f"local_original_{families}_{model_tag}_FastSigmf_stagecost_"
        f"{output_parameter_tag(args, devices)}_sweep_summary"
    )


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
        *PROFILE_NUMERIC_FIELDS,
        *PROFILE_STRING_FIELDS,
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
    profile_schema: dict[str, Any] | None = None
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    experiment_start = time.perf_counter()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        writer.writeheader()
        for meta_path, data_path in pairs:
            sample_id = meta_path.name.replace(".sigmf-meta", "")
            t_total = time.perf_counter()

            x, preprocess_profile = profiled_preprocess_sigmf_to_tensor(meta_path, data_path, preprocessor)
            read_convert_sec = profile_stage_total(preprocess_profile, PREPROCESS_STAGES)

            raw_output, model_profile, sample_profile_schema = profiled_model_forward(args.model_name, model, x, device)
            if profile_schema is None:
                profile_schema = sample_profile_schema
            inference_sec = profile_stage_total(model_profile, MODEL_STAGES)

            quality_error = ""
            postprocess_profile = empty_stage_profile(POSTPROCESS_STAGES)
            try:
                t0 = time.perf_counter()
                raw_output_cpu = move_to_cpu(raw_output)
                detections, postprocess_backend = postprocess_raw_output(
                    args.model_name,
                    raw_output_cpu,
                    args.conf_threshold,
                    args.iou_threshold,
                )
                postprocess_profile["post_raw_decode_sec"] = time.perf_counter() - t0
                postprocess_profile["post_raw_decode_bytes"] = tensor_bytes(detections)

                t0 = time.perf_counter()
                labels = load_yolo_label_tensor(label_path_for_sample(args.yolo_root, args.split, sample_id))
                postprocess_profile["post_load_label_sec"] = time.perf_counter() - t0
                postprocess_profile["post_load_label_bytes"] = tensor_bytes(labels)

                t0 = time.perf_counter()
                sample_metrics = metrics.add(detections, labels)
                postprocess_profile["post_detection_metric_sec"] = time.perf_counter() - t0
                postprocess_profile["post_detection_metric_bytes"] = tensor_bytes(detections) + tensor_bytes(labels)

                t0 = time.perf_counter()
                sample_overlap_metrics = overlap_metrics.add(
                    pred_overlap=overlap_state_from_detections(detections),
                    gt_overlap=overlap_state_from_labels(labels),
                )
                postprocess_profile["post_overlap_metric_sec"] = time.perf_counter() - t0
                postprocess_profile["post_overlap_metric_bytes"] = tensor_bytes(detections) + tensor_bytes(labels)
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
            postprocess_sec = profile_stage_total(postprocess_profile, POSTPROCESS_STAGES)
            completed_at_sec = time.perf_counter() - experiment_start

            row = {
                "sample_id": sample_id,
                "read_convert_sec": read_convert_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "end_to_end_sec": time.perf_counter() - t_total,
                "completed_at_sec": completed_at_sec,
                **empty_profile(),
                **preprocess_profile,
                **model_profile,
                **postprocess_profile,
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

    summary = {
        "experiment": {
            "execution": "local",
            "architecture": "original",
            "model_name": args.model_name,
            "model_path": str(args.model_path),
            "device": str(device),
            "run_location": args.run_location,
            "split": args.split,
            "batch_size": 1,
            "img_size": IMG_SIZE,
            "max_samples": normalize_max_samples(args.max_samples),
            "sample_count": len(rows),
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
        },
        "detection": metrics.summary(),
        "overlap": overlap_metrics.summary(),
        "profile_schema": profile_schema or {},
    }
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
    row = {
        "model_name": args.model_name,
        "device": args.device,
        "split": args.split,
        "max_samples": normalize_max_samples(args.max_samples),
        "execution": "local",
        "architecture": "original",
        "run_location": args.run_location,
        "batch_size": 1,
        "img_size": IMG_SIZE,
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
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
    for key in PROFILE_NUMERIC_FIELDS:
        row[f"mean_{key}"] = mean_numeric(rows, key)
    return row


def write_sweep_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sweep_output_stem(args)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
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
                write_sweep_summary(summary_rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential local CBRS SigMF stage cost test")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep_models", default=",".join(DEFAULT_SWEEP_MODELS))
    parser.add_argument("--sweep_devices", default=",".join(DEFAULT_SWEEP_DEVICES))
    parser.add_argument("--raw_root", type=Path, default=RAW_SIGMF_ROOT)
    parser.add_argument("--yolo_root", type=Path, default=YOLO_SPLIT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument(
        "--max_samples", type=int, default=MAX_SAMPLES,
        help="Sample limit; 0 or negative means all samples. Overrides the global MAX_SAMPLES.",
    )
    parser.add_argument("--run_location", default="local")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--quality_json", type=Path, default=None)
    parser.add_argument("--plot_png", type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou_threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.max_samples = normalize_max_samples(args.max_samples)
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
