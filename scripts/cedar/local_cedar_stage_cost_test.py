"""
Sequential local stage cost test for AsymYOLO (SigYOLO) checkpoints on raw SigMF samples.

CEDAR-YOLO in the paper is called SigYOLO / AsymYOLO (sig_yolo_* runs) in the code.

Mirrors local_sigmf_stage_cost_test.py — same CSV/JSON/PNG outputs, same sweep flow,
same preprocess/postprocess/metric code — but adapted to AsymYOLO models:

    raw SigMF pair
        -> FastSigMF preprocess (7-stage profiled, same as baseline)
        -> AsymYOLO forward (split per backbone stage + sppf + psa + head)
        -> asym_non_max_suppression
        -> detection / overlap metrics

AsymYOLO architecture differences vs Ultralytics YOLO/RT-DETR:
  - Single-scale: no FPN/PAN neck, no DETR encoder/decoder
  - Stages: variable count (3-7) depending on fm_freq/fm_time
  - Tail: backbone -> sppf -> psa(Identity for v8/v26, real PSA for v11) -> head
  - Head is split into three sub-stages for finer cost analysis:
      head_box     : cv2 branch (box raw, no_box channels)
      head_cls     : cv3 branch (cls raw, nc channels)
      head_decode  : DFL + anchor decode + sigmoid + concat → final y
    Two extra boundary tensors recorded: box_to_decode, cls_to_decode.
  - Boundary tensors are single tensors (not [P3, P4, P5] triples)

Discovers all sig_yolo_*/weights/best.pt (fallback last.pt) under trained_models/
unless --sweep_models is given explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import csv
import gc
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cbrs.cedar_yolo import asym_non_max_suppression, build_asym_yolo  # noqa: E402
from cbrs.sigmf_fast_tensor_preprocessor import (                       # noqa: E402
    FastSigMFTensorPreprocessor,
    tensor_from_rgb_array,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths / defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_baseline_results_by_stage_cost_asym_yolo"
RAW_SIGMF_ROOT     = PROJECT_ROOT / "raw_sigmf_split"
YOLO_SPLIT_ROOT    = PROJECT_ROOT / "yolo_split"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

IMG_SIZE       = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
CLASS_NAMES    = {0: "LTE", 1: "Radar"}

# Safe upper bound for input=640 (smallest valid fm gives 7 downsample steps).
MAX_BACKBONE_STAGES = 7

# Default sweep set: v8 nano/small/x at the canonical f20×t20 c100 configuration
# (stride 32/32 — matches official YOLOv8 P5 placement; reg_max = full coverage).
DEFAULT_SWEEP_MODELS = [
    "sig_yolo_v8n_f20t20_c100",
    "sig_yolo_v8s_f20t20_c100",
    "sig_yolo_v8x_f20t20_c100",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage names
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESS_STAGES = [
    "pre_read_iq",
    "pre_spectrogram",
    "pre_fftshift_log",
    "pre_percentile_clip",
    "pre_canvas_render",
    "pre_canvas_to_rgb",
    "pre_to_tensor",
]
MODEL_STAGES = (
    [f"model_stage_{i}" for i in range(MAX_BACKBONE_STAGES)]
    + ["model_sppf", "model_psa",
       "model_head_box", "model_head_cls", "model_head_decode"]
)
POSTPROCESS_STAGES = [
    "post_raw_decode",
    "post_load_label",
    "post_detection_metric",
    "post_overlap_metric",
]

PROFILE_NUMERIC_FIELDS = (
    [f"{s}_{sfx}" for s in PREPROCESS_STAGES for sfx in ("sec", "bytes")]
    + [f"{s}_{sfx}" for s in MODEL_STAGES for sfx in ("sec", "bytes")]
    + [
        "model_backbone_to_sppf_bytes",
        "model_sppf_to_psa_bytes",
        "model_psa_to_head_bytes",
        "model_box_to_decode_bytes",
        "model_cls_to_decode_bytes",
    ]
    + [f"{s}_{sfx}" for s in POSTPROCESS_STAGES for sfx in ("sec", "bytes")]
)
PROFILE_STRING_FIELDS = [
    "model_backbone_to_sppf_tensors",
    "model_sppf_to_psa_tensors",
    "model_psa_to_head_tensors",
    "model_box_to_decode_tensors",
    "model_cls_to_decode_tensors",
]


# ─────────────────────────────────────────────────────────────────────────────
# Model discovery + loading
# ─────────────────────────────────────────────────────────────────────────────

_RUN_RE = re.compile(
    r"sig_yolo_(?P<variant>v\d+)(?P<size>[nsmxl])"
    r"_f(?P<fm_freq>\d+)t(?P<fm_time>\d+)"
    r"_c(?P<cpct>\d+)$"
)


def parse_run_name(run_name: str) -> dict | None:
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


def discover_asym_models(models_dir: Path = TRAINED_MODELS_DIR) -> list[str]:
    """Return sorted list of sig_yolo_* run names that have a best.pt or last.pt."""
    if not models_dir.exists():
        return []
    out: list[str] = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("sig_yolo_"):
            continue
        weights = d / "weights"
        if (weights / "best.pt").exists() or (weights / "last.pt").exists():
            out.append(d.name)
    return out


def resolve_model_name(model_name: str) -> str:
    key = str(model_name).strip()
    if not key.startswith("sig_yolo_") or not _RUN_RE.match(key):
        raise ValueError(
            f"Unsupported model_name={model_name!r}; "
            f"expected pattern sig_yolo_v<N><n|s|m|l|x>_f<FF>t<FT>_c<CPCT>"
        )
    return key


def model_path_for_name(model_name: str) -> Path:
    weights_dir = TRAINED_MODELS_DIR / model_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    return best if best.exists() else last


def load_detection_model(model_name: str, model_path: Path, device: torch.device):
    """Load an AsymYOLO checkpoint via build_asym_yolo + state_dict."""
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)

    variant      = ckpt.get("variant")
    size         = ckpt.get("size")
    fm_freq      = ckpt.get("fm_freq")
    fm_time      = ckpt.get("fm_time")
    reg_max_freq = ckpt.get("reg_max_freq")
    reg_max_time = ckpt.get("reg_max_time")
    nc           = ckpt.get("nc", 2)
    img_size     = int(ckpt.get("img_size", IMG_SIZE))

    if None in (variant, size, fm_freq, fm_time, reg_max_freq, reg_max_time):
        parsed = parse_run_name(model_name)
        if parsed is None:
            raise RuntimeError(f"Cannot determine arch params for {model_name!r}")
        variant      = variant      or parsed["variant"]
        size         = size         or parsed["size"]
        fm_freq      = fm_freq      or parsed["fm_freq"]
        fm_time      = fm_time      or parsed["fm_time"]
        reg_max_freq = reg_max_freq or parsed["reg_max_freq"]
        reg_max_time = reg_max_time or parsed["reg_max_time"]

    model = build_asym_yolo(
        variant=variant, size=size, nc=int(nc),
        fm_freq=int(fm_freq), fm_time=int(fm_time),
        reg_max_freq=int(reg_max_freq), reg_max_time=int(reg_max_time),
        input_w=img_size, input_h=img_size,
    )
    state = ckpt.get("ema", ckpt.get("model"))
    if state is None:
        raise RuntimeError(f"No weights in checkpoint {model_path}")
    model.load_state_dict(state, strict=True)
    model.to(device).float().eval()
    # Stash arch info for downstream output
    model._asym_arch_info = {
        "variant": variant, "size": size, "nc": int(nc),
        "fm_freq": int(fm_freq), "fm_time": int(fm_time),
        "reg_max_freq": int(reg_max_freq), "reg_max_time": int(reg_max_time),
        "img_size": img_size, "n_backbone_stages": len(list(model.backbone.children())),
        "has_real_psa": not isinstance(model.psa, torch.nn.Identity),
    }
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Profile bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

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
            "model_backbone_to_sppf_bytes":   0.0,
            "model_sppf_to_psa_bytes":        0.0,
            "model_psa_to_head_bytes":        0.0,
            "model_box_to_decode_bytes":      0.0,
            "model_cls_to_decode_bytes":      0.0,
            "model_backbone_to_sppf_tensors": "",
            "model_sppf_to_psa_tensors":      "",
            "model_psa_to_head_tensors":      "",
            "model_box_to_decode_tensors":    "",
            "model_cls_to_decode_tensors":    "",
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mostly verbatim from local_sigmf_stage_cost_test.py)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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
            x1 = (x_center - width  / 2.0) * IMG_SIZE
            y1 = (y_center - height / 2.0) * IMG_SIZE
            x2 = (x_center + width  / 2.0) * IMG_SIZE
            y2 = (y_center + height / 2.0) * IMG_SIZE
            labels.append([cls, x1, y1, x2, y2])
    if not labels:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(labels, dtype=torch.float32)


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


def profile_stage_total(profile: dict[str, Any], stages: list[str]) -> float:
    return float(sum(float(profile.get(f"{stage}_sec", 0.0) or 0.0) for stage in stages))


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess (identical to baseline)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Model forward (per-stage profiling, AsymYOLO-specific)
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_sync(value: Any) -> None:
    if not torch.cuda.is_available():
        return
    ft = first_tensor_in(value)
    if ft is not None and ft.is_cuda:
        torch.cuda.synchronize(ft.device)


def run_profiled_stage_single(
    profile: dict[str, Any],
    stage_name: str,
    module: torch.nn.Module,
    x: Any,
) -> Any:
    """Time a single nn.Module call, record sec + output bytes."""
    _cuda_sync(x)
    t0 = time.perf_counter()
    out = module(x)
    _cuda_sync(out)
    profile[f"{stage_name}_sec"]   = time.perf_counter() - t0
    profile[f"{stage_name}_bytes"] = tensor_bytes(out)
    return out


def _boundary_meta(name: str, value: Any) -> tuple[int, str]:
    shapes = tensor_shapes(value, limit=4)
    total  = sum(int(s["bytes"]) for s in shapes)
    meta   = [{"name": name, **s} for s in shapes]
    return total, json.dumps(meta)


def profiled_model_forward(
    model_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """
    Iterate the AsymYOLO modules in execution order:
        backbone.stages[0..N-1]  ->  sppf  ->  psa  ->  head

    Boundary tensors recorded:
        backbone_to_sppf, sppf_to_psa, psa_to_head
    """
    profile = empty_model_profile()
    arch_info = getattr(model, "_asym_arch_info", {})
    schema: dict[str, Any] = {
        "model_name":       model_name,
        "arch":             arch_info,
        "stage_ranges":     {},
        "boundary_tensors": {},
    }

    current: Any = x.to(device)
    backbone_stages = list(model.backbone.children())

    with torch.inference_mode():
        # Backbone stages — one profiled span each
        for idx, stage_module in enumerate(backbone_stages):
            stage_name = f"model_stage_{idx}"
            current = run_profiled_stage_single(profile, stage_name, stage_module, current)
            schema["stage_ranges"][stage_name] = [idx, idx + 1]

        # backbone -> sppf boundary  (single tensor)
        b_bytes, b_meta = _boundary_meta("backbone_out", current)
        profile["model_backbone_to_sppf_bytes"]   = b_bytes
        profile["model_backbone_to_sppf_tensors"] = b_meta
        schema["boundary_tensors"]["backbone_to_sppf"] = json.loads(b_meta)

        # SPPF
        current = run_profiled_stage_single(profile, "model_sppf", model.sppf, current)

        # sppf -> psa boundary
        b_bytes, b_meta = _boundary_meta("sppf_out", current)
        profile["model_sppf_to_psa_bytes"]   = b_bytes
        profile["model_sppf_to_psa_tensors"] = b_meta
        schema["boundary_tensors"]["sppf_to_psa"] = json.loads(b_meta)

        # PSA (nn.Identity for v8/v26 — still timed; observed cost ≈ 0)
        current = run_profiled_stage_single(profile, "model_psa", model.psa, current)

        # psa -> head boundary (same tensor flows into BOTH box and cls branches)
        b_bytes, b_meta = _boundary_meta("psa_out", current)
        profile["model_psa_to_head_bytes"]   = b_bytes
        profile["model_psa_to_head_tensors"] = b_meta
        schema["boundary_tensors"]["psa_to_head"] = json.loads(b_meta)

        # ── Head split into three sub-stages ──────────────────────────────────
        #
        # forward logic (mirrors cbrs/cedar_yolo.py :: AsymDetect.forward):
        #   box_raw = cv2(x).view(B, no_box, A)        ← model_head_box
        #   cls_raw = cv3(x).view(B, nc,     A)        ← model_head_cls
        #   boxes   = _decode(box_raw, H, W).permute(0,2,1)
        #   y       = cat([boxes, cls_raw.sigmoid()], dim=1)   ← model_head_decode
        #
        head      = model.head
        head_in   = current
        B, _, H, W = head_in.shape
        A = H * W

        # Stage: head_box — cv2 branch (factored convs + 1×1 → no_box channels)
        _cuda_sync(head_in)
        t0 = time.perf_counter()
        box_raw = head.cv2(head_in).view(B, head.no_box, A)
        _cuda_sync(box_raw)
        profile["model_head_box_sec"]   = time.perf_counter() - t0
        profile["model_head_box_bytes"] = tensor_bytes(box_raw)
        schema["stage_ranges"]["model_head_box"] = ["head.cv2", "view"]

        # Boundary: box_raw → decode
        b_bytes, b_meta = _boundary_meta("box_raw", box_raw)
        profile["model_box_to_decode_bytes"]   = b_bytes
        profile["model_box_to_decode_tensors"] = b_meta
        schema["boundary_tensors"]["box_to_decode"] = json.loads(b_meta)

        # Stage: head_cls — cv3 branch (factored convs + 1×1 → nc channels)
        _cuda_sync(head_in)
        t0 = time.perf_counter()
        cls_raw = head.cv3(head_in).view(B, head.nc, A)
        _cuda_sync(cls_raw)
        profile["model_head_cls_sec"]   = time.perf_counter() - t0
        profile["model_head_cls_bytes"] = tensor_bytes(cls_raw)
        schema["stage_ranges"]["model_head_cls"] = ["head.cv3", "view"]

        # Boundary: cls_raw → decode
        b_bytes, b_meta = _boundary_meta("cls_raw", cls_raw)
        profile["model_cls_to_decode_bytes"]   = b_bytes
        profile["model_cls_to_decode_tensors"] = b_meta
        schema["boundary_tensors"]["cls_to_decode"] = json.loads(b_meta)

        # Stage: head_decode — DFL decode + anchor mesh + sigmoid + concat
        _cuda_sync(box_raw)
        t0 = time.perf_counter()
        boxes    = head._decode(box_raw, H, W).permute(0, 2, 1)   # [B, 4, A]
        cls_prob = cls_raw.sigmoid()                              # [B, nc, A]
        y = torch.cat([boxes, cls_prob], dim=1)                   # [B, 4+nc, A]
        # current matches the tuple shape that head.forward() returns in eval mode,
        # so downstream postprocess code stays identical.
        current = (y, (box_raw, cls_raw))
        _cuda_sync(y)
        profile["model_head_decode_sec"]   = time.perf_counter() - t0
        # bytes counts only the NEW tensor produced (y); box_raw/cls_raw are
        # passed-through references, not freshly generated data.
        profile["model_head_decode_bytes"] = tensor_bytes(y)
        schema["stage_ranges"]["model_head_decode"] = ["head._decode", "sigmoid+concat"]

    return current, profile, schema


# ─────────────────────────────────────────────────────────────────────────────
# Postprocess — AsymYOLO NMS
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_raw_output(
    model_name: str,
    raw_output: Any,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[torch.Tensor, str]:
    """
    AsymYOLO head returns (y_inference, y_train).
    y_inference shape: [B, 4+nc, A] in pixel coordinates (xyxy via DFL decode).
    asym_non_max_suppression handles class-aware NMS and returns List[Tensor[N, 6]].
    """
    _ = model_name
    if isinstance(raw_output, (tuple, list)):
        y = raw_output[0]
    else:
        y = raw_output

    if not torch.is_tensor(y):
        return torch.zeros((0, 6), dtype=torch.float32), "local_asym_nms_failed"

    y = y if y.ndim == 3 else y.unsqueeze(0)
    dets_per_image = asym_non_max_suppression(
        y,
        conf_thres=float(conf_threshold),
        iou_thres=float(iou_threshold),
        max_det=300,
    )
    if not dets_per_image:
        return torch.zeros((0, 6), dtype=torch.float32), "local_asym_nms"
    det = dets_per_image[0].detach().cpu().float()
    return det if det.numel() else torch.zeros((0, 6), dtype=torch.float32), "local_asym_nms"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — verbatim from baseline
# ─────────────────────────────────────────────────────────────────────────────

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


def detection_correct_matrix(detections: torch.Tensor, labels: torch.Tensor, iou_thresholds: np.ndarray) -> np.ndarray:
    correct = np.zeros((detections.shape[0], len(iou_thresholds)), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    ious        = box_iou_xyxy(detections[:, :4].float(), labels[:, 1:5].float())
    class_match = detections[:, 5:6].int() == labels[:, 0].int().view(1, -1)
    for ti, threshold in enumerate(iou_thresholds):
        det_idx, lbl_idx = torch.where((ious >= float(threshold)) & class_match)
        pairs = [(int(d), int(l), float(ious[d, l])) for d, l in zip(det_idx, lbl_idx)]
        pairs.sort(key=lambda item: item[2], reverse=True)
        used_d: set[int] = set()
        used_l: set[int] = set()
        for d, l, _ in pairs:
            if d in used_d or l in used_l:
                continue
            correct[d, ti] = True
            used_d.add(d); used_l.add(l)
    return correct


def has_frequency_overlap_xyxy(boxes: torch.Tensor, classes: torch.Tensor) -> bool:
    lte_boxes   = boxes[classes.int() == 0]
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
        self.conf:    list[np.ndarray] = []
        self.pred_cls:   list[np.ndarray] = []
        self.target_cls: list[np.ndarray] = []

    def add(self, detections: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        detections = detections.detach().cpu()
        labels     = labels.detach().cpu()
        correct    = detection_correct_matrix(detections, labels, self.iouv)
        conf       = detections[:, 4].numpy() if detections.shape[0] else np.zeros((0,), dtype=float)
        pred_cls   = detections[:, 5].numpy() if detections.shape[0] else np.zeros((0,), dtype=float)
        target_cls = labels[:, 0].numpy()    if labels.shape[0]     else np.zeros((0,), dtype=float)
        self.correct.append(correct)
        self.conf.append(conf)
        self.pred_cls.append(pred_cls)
        self.target_cls.append(target_cls)
        tp50  = int(correct[:, 0].sum()) if correct.size else 0
        pcnt  = int(detections.shape[0])
        gcnt  = int(labels.shape[0])
        return {
            "gt_count":     gcnt,
            "pred_count":   pcnt,
            "tp50":         tp50,
            "fp50":         pcnt - tp50,
            "fn50":         gcnt - tp50,
            "precision50":  tp50 / pcnt if pcnt > 0 else 0.0,
            "recall50":     tp50 / gcnt if gcnt > 0 else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        target_cls = np.concatenate(self.target_cls) if self.target_cls else np.zeros((0,), dtype=float)
        correct    = np.concatenate(self.correct, axis=0) if self.correct else np.zeros((0, len(self.iouv)), dtype=bool)
        conf       = np.concatenate(self.conf)     if self.conf     else np.zeros((0,), dtype=float)
        pred_cls   = np.concatenate(self.pred_cls) if self.pred_cls else np.zeros((0,), dtype=float)
        total_gt   = int(target_cls.shape[0])
        total_pred = int(pred_cls.shape[0])
        tp50       = int(correct[:, 0].sum()) if correct.size else 0
        ap = np.zeros((len(CLASS_NAMES), len(self.iouv)), dtype=float)
        for class_id in CLASS_NAMES:
            class_mask = pred_cls == class_id
            n_gt   = int((target_cls == class_id).sum())
            n_pred = int(class_mask.sum())
            if n_gt == 0 or n_pred == 0:
                continue
            class_correct = correct[class_mask][np.argsort(-conf[class_mask])]
            for ti in range(len(self.iouv)):
                tp = class_correct[:, ti].astype(float)
                fp = 1.0 - tp
                tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
                ap[class_id, ti] = compute_ap(
                    tp_cum / (n_gt + 1e-16),
                    tp_cum / (tp_cum + fp_cum + 1e-16),
                )
        return {
            "model":            self.model_name,
            "classes":          CLASS_NAMES,
            "iou_thresholds":   [float(v) for v in self.iouv],
            "image_count":      len(self.target_cls),
            "target_count":     total_gt,
            "prediction_count": total_pred,
            "precision":        float(tp50 / total_pred) if total_pred > 0 else 0.0,
            "recall":           float(tp50 / total_gt)   if total_gt   > 0 else 0.0,
            "map50":            float(ap[:, 0].mean()),
            "map50_95":         float(ap.mean()),
            "ap50_by_class":    {CLASS_NAMES[c]: float(ap[c, 0]) for c in CLASS_NAMES},
        }


class OverlapMetricAccumulator:
    def __init__(self) -> None:
        self.tp = self.tn = self.fp = self.fn = 0

    def add(self, pred_overlap: bool, gt_overlap: bool) -> dict[str, Any]:
        tp = int(gt_overlap and pred_overlap)
        tn = int((not gt_overlap) and (not pred_overlap))
        fp = int((not gt_overlap) and pred_overlap)
        fn = int(gt_overlap and (not pred_overlap))
        self.tp += tp; self.tn += tn; self.fp += fp; self.fn += fn
        return {
            "gt_overlap":   int(gt_overlap),
            "pred_overlap": int(pred_overlap),
            "overlap_tp":   tp,
            "overlap_tn":   tn,
            "overlap_fp":   fp,
            "overlap_fn":   fn,
        }

    def summary(self) -> dict[str, Any]:
        total = self.tp + self.tn + self.fp + self.fn
        prec  = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec   = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return {
            "TP": self.tp, "TN": self.tn, "FP": self.fp, "FN": self.fn,
            "overlap_accuracy":  (self.tp + self.tn) / total if total else 0.0,
            "overlap_precision": prec,
            "overlap_recall":    rec,
            "overlap_f1":        2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Output paths + CSV fields
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

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
    fig.suptitle(f"{summary.get('detection', {}).get('model', 'model')} local AsymYOLO baseline ({len(rows)} samples)")
    axes[0].bar(latency_labels, latency_values, color="#4C78A8")
    axes[0].set_title("Mean Sequential Latency"); axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[1].bar(quality_labels, quality_values, color="#E45756")
    axes[1].set_title("Final Quality"); axes[1].set_ylim(0.0, 1.0); axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", labelrotation=20)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
        for patch in ax.patches:
            v = patch.get_height()
            ax.text(patch.get_x() + patch.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[baseline][plot] summary plot written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main per-model runner
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model  = load_detection_model(args.model_name, args.model_path, device)
    preprocessor = FastSigMFTensorPreprocessor()
    pairs  = collect_sigmf_pairs(args.raw_root, args.split, args.max_samples)
    print(f"[baseline] model={args.model_name} samples={len(pairs)} device={device} "
          f"arch={model._asym_arch_info}")

    metrics         = DetectionMetricAccumulator(args.model_name)
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

            raw_output, model_profile, sample_profile_schema = profiled_model_forward(
                args.model_name, model, x.unsqueeze(0) if x.ndim == 3 else x, device,
            )
            if profile_schema is None:
                profile_schema = sample_profile_schema
            inference_sec = profile_stage_total(model_profile, MODEL_STAGES)

            quality_error = ""
            postprocess_profile = empty_stage_profile(POSTPROCESS_STAGES)
            try:
                t0 = time.perf_counter()
                raw_output_cpu = move_to_cpu(raw_output)
                detections, postprocess_backend = postprocess_raw_output(
                    args.model_name, raw_output_cpu, args.conf_threshold, args.iou_threshold,
                )
                postprocess_profile["post_raw_decode_sec"]   = time.perf_counter() - t0
                postprocess_profile["post_raw_decode_bytes"] = tensor_bytes(detections)

                t0 = time.perf_counter()
                labels = load_yolo_label_tensor(label_path_for_sample(args.yolo_root, args.split, sample_id))
                postprocess_profile["post_load_label_sec"]   = time.perf_counter() - t0
                postprocess_profile["post_load_label_bytes"] = tensor_bytes(labels)

                t0 = time.perf_counter()
                sample_metrics = metrics.add(detections, labels)
                postprocess_profile["post_detection_metric_sec"]   = time.perf_counter() - t0
                postprocess_profile["post_detection_metric_bytes"] = tensor_bytes(detections) + tensor_bytes(labels)

                t0 = time.perf_counter()
                sample_overlap_metrics = overlap_metrics.add(
                    pred_overlap=overlap_state_from_detections(detections),
                    gt_overlap=overlap_state_from_labels(labels),
                )
                postprocess_profile["post_overlap_metric_sec"]   = time.perf_counter() - t0
                postprocess_profile["post_overlap_metric_bytes"] = tensor_bytes(detections) + tensor_bytes(labels)
            except Exception as exc:
                raw_output_cpu = move_to_cpu(raw_output)
                postprocess_backend = "failed"
                quality_error = repr(exc)
                sample_metrics = {
                    "gt_count": None, "pred_count": None,
                    "tp50": None, "fp50": None, "fn50": None,
                    "precision50": None, "recall50": None,
                }
                sample_overlap_metrics = {
                    "gt_overlap": None, "pred_overlap": None,
                    "overlap_tp": None, "overlap_tn": None,
                    "overlap_fp": None, "overlap_fn": None,
                }
            postprocess_sec  = profile_stage_total(postprocess_profile, POSTPROCESS_STAGES)
            completed_at_sec = time.perf_counter() - experiment_start

            row = {
                "sample_id":         sample_id,
                "read_convert_sec":  read_convert_sec,
                "inference_sec":     inference_sec,
                "postprocess_sec":   postprocess_sec,
                "end_to_end_sec":    time.perf_counter() - t_total,
                "completed_at_sec":  completed_at_sec,
                **empty_profile(),
                **preprocess_profile,
                **model_profile,
                **postprocess_profile,
                "raw_output_shapes": json.dumps(tensor_shapes(raw_output_cpu)),
                "postprocess_backend": postprocess_backend,
                **sample_metrics,
                **sample_overlap_metrics,
                "quality_error":     quality_error,
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
        "detection":      metrics.summary(),
        "overlap":        overlap_metrics.summary(),
        "profile_schema": profile_schema or {},
        "arch":           getattr(model, "_asym_arch_info", {}),
    }
    args.quality_json.parent.mkdir(parents=True, exist_ok=True)
    args.quality_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[baseline] csv written to {args.output_csv}")
    print(f"[baseline] quality summary written to {args.quality_json}")
    render_plot(rows, summary, args.plot_png)
    return {
        "summary":           summary,
        "rows":              rows,
        "csv_path":          str(args.output_csv),
        "quality_json_path": str(args.quality_json),
        "plot_png_path":     str(args.plot_png),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def summarize_run_result(args: argparse.Namespace, result: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    rows      = result.get("rows", [])
    summary   = result.get("summary", {})
    detection = summary.get("detection", {})
    overlap   = summary.get("overlap", {})
    arch      = summary.get("arch", {}) or {}
    completed = [float(row["completed_at_sec"]) for row in rows if row.get("completed_at_sec") not in (None, "")]
    elapsed   = max(completed) if completed else 0.0
    image_count = int(detection.get("image_count", 0) or 0)
    row = {
        "model_name":         args.model_name,
        "variant":            arch.get("variant"),
        "size":               arch.get("size"),
        "fm_freq":            arch.get("fm_freq"),
        "fm_time":            arch.get("fm_time"),
        "reg_max_freq":       arch.get("reg_max_freq"),
        "reg_max_time":       arch.get("reg_max_time"),
        "n_backbone_stages":  arch.get("n_backbone_stages"),
        "has_real_psa":       arch.get("has_real_psa"),
        "device":             args.device,
        "split":              args.split,
        "max_samples":        args.max_samples,
        "csv_path":           result.get("csv_path"),
        "quality_json_path":  result.get("quality_json_path"),
        "plot_png_path":      result.get("plot_png_path"),
        "image_count":        image_count,
        "target_count":       detection.get("target_count"),
        "prediction_count":   detection.get("prediction_count"),
        "map50":              detection.get("map50"),
        "map50_95":           detection.get("map50_95"),
        "det_precision":      detection.get("precision"),
        "det_recall":         detection.get("recall"),
        "TP":                 overlap.get("TP"),
        "TN":                 overlap.get("TN"),
        "FP":                 overlap.get("FP"),
        "FN":                 overlap.get("FN"),
        "overlap_accuracy":   overlap.get("overlap_accuracy"),
        "overlap_precision":  overlap.get("overlap_precision"),
        "overlap_recall":     overlap.get("overlap_recall"),
        "overlap_f1":         overlap.get("overlap_f1"),
        "mean_read_convert_sec": mean_numeric(rows, "read_convert_sec"),
        "mean_inference_sec":    mean_numeric(rows, "inference_sec"),
        "mean_postprocess_sec":  mean_numeric(rows, "postprocess_sec"),
        "mean_end_to_end_sec":   mean_numeric(rows, "end_to_end_sec"),
        "throughput_samples_per_sec": image_count / elapsed if elapsed > 0 else 0.0,
        "status":             status,
        "error":              error,
    }
    for key in PROFILE_NUMERIC_FIELDS:
        row[f"mean_{key}"] = mean_numeric(rows, key)
    return row


def write_sweep_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep_summary.json"
    csv_path  = output_dir / "sweep_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else ["model_name", "device", "status", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] summary csv written to {csv_path}")
    print(f"[sweep] summary json written to {json_path}")


def run_sweep(args: argparse.Namespace) -> None:
    if args.sweep_models is None or not args.sweep_models.strip():
        models = discover_asym_models(TRAINED_MODELS_DIR)
        if not models:
            print(f"[sweep] No sig_yolo_* checkpoints found under {TRAINED_MODELS_DIR}")
            return
        print(f"[sweep] auto-discovered {len(models)} model(s)")
    else:
        models = [resolve_model_name(name) for name in parse_csv_list(args.sweep_models)]
    devices = parse_csv_list(args.sweep_devices)
    summary_rows: list[dict[str, Any]] = []

    for device_name in devices:
        for model_name in models:
            run_args = deepcopy(args)
            run_args.sweep        = False
            run_args.model_name   = model_name
            run_args.model_path   = model_path_for_name(model_name)
            run_args.device       = device_name
            run_args.output_csv   = None
            run_args.quality_json = None
            run_args.plot_png     = None
            resolve_output_paths(run_args)

            print(f"[sweep] starting model={model_name} device={device_name}")
            cleanup_runtime()
            if device_name.startswith("cuda") and not torch.cuda.is_available():
                result = {"rows": [], "summary": {"detection": {"image_count": 0}, "overlap": {}, "arch": {}}}
                summary_rows.append(summarize_run_result(run_args, result, "skipped", "cuda unavailable"))
                print(f"[sweep] skipped model={model_name} device={device_name}: cuda unavailable")
                continue

            try:
                result = run_baseline(run_args)
                summary_rows.append(summarize_run_result(run_args, result, "ok"))
            except Exception as exc:
                result = {"rows": [], "summary": {"detection": {"image_count": 0}, "overlap": {}, "arch": {}}}
                summary_rows.append(summarize_run_result(run_args, result, "failed", repr(exc)))
                print(f"[sweep] failed model={model_name} device={device_name}: {exc!r}")
            finally:
                cleanup_runtime()
                write_sweep_summary(summary_rows, args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    global TRAINED_MODELS_DIR
    parser = argparse.ArgumentParser(
        description="Sequential local CBRS SigMF stage cost test for AsymYOLO checkpoints"
    )
    parser.add_argument("--model_name", default=None,
                        help="sig_yolo_* run name; required when --no-sweep")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--models_dir", type=Path, default=TRAINED_MODELS_DIR,
                        help="Directory containing sig_yolo_*/weights/best.pt "
                             "(use trained_models_mdi for the Jetson MDI local baseline)")
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep_models", default=",".join(DEFAULT_SWEEP_MODELS),
                        help="comma-separated sig_yolo_* names; "
                             "default = v8 n/s/x at f20t20_c100; "
                             "pass empty string ('') for auto-discover all checkpoints")
    parser.add_argument("--sweep_devices", default="cpu,cuda")
    parser.add_argument("--raw_root",  type=Path, default=RAW_SIGMF_ROOT)
    parser.add_argument("--yolo_root", type=Path, default=YOLO_SPLIT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_location", default="local")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_csv",  type=Path, default=None)
    parser.add_argument("--quality_json", type=Path, default=None)
    parser.add_argument("--plot_png",    type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou_threshold",  type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    TRAINED_MODELS_DIR = args.models_dir
    if not args.sweep:
        if args.model_name is None:
            parser.error("--model_name is required when --no-sweep")
        args.model_name = resolve_model_name(args.model_name)
        if args.model_path is None:
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
