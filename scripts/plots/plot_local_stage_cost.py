"""
Plot local FastSigmf stage-cost results.

One output figure is generated per model. Each figure contains:
    1. average module delay at the allowed cutpoints
    2. minimal module-boundary payload size when modules are not split
    3. estimated module-boundary transfer time at selected link rates
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATES_MBPS = [25.0, 50.0, 150.0]
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "local_baseline_results_by_stage_cost_FastSigmf_cpu",
    PROJECT_ROOT / "local_test" / "local_baseline_results_by_stage_cost_FastSigmf_cpu",
    PROJECT_ROOT / "local_baseline_results_by_stage_cost",
]

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
ALL_STAGES = PREPROCESS_STAGES + MODEL_STAGES + POSTPROCESS_STAGES

YOLO8_MODULE_BOUNDARIES = [
    {
        "name": "preprocess -> backbone",
        "logical": "image",
        "payload_fields": ["pre_to_tensor_bytes"],
    },
    {
        "name": "backbone -> neck",
        "logical": "x=layer9, cache=[4,6]",
        "total_field": "model_backbone_to_neck_bytes",
        "fallback_fields": ["model_p3_bytes", "model_p4_bytes", "model_p5_bytes"],
    },
    {
        "name": "neck -> head",
        "logical": "cache=[15,18,21]",
        "total_field": "model_neck_bytes",
    },
    {
        "name": "head -> A",
        "logical": "detections",
        "payload_fields": ["post_raw_decode_bytes"],
    },
]
YOLO11_26_MODULE_BOUNDARIES = [
    {
        "name": "preprocess -> backbone",
        "logical": "image",
        "payload_fields": ["pre_to_tensor_bytes"],
    },
    {
        "name": "backbone -> neck",
        "logical": "x=layer10, cache=[4,6]",
        "total_field": "model_backbone_to_neck_bytes",
        "fallback_fields": ["model_p3_bytes", "model_p4_bytes", "model_p5_bytes"],
    },
    {
        "name": "neck -> head",
        "logical": "cache=[16,19,22]",
        "total_field": "model_neck_bytes",
    },
    {
        "name": "head -> A",
        "logical": "detections",
        "payload_fields": ["post_raw_decode_bytes"],
    },
]
RTDETR_MODULE_BOUNDARIES = [
    {
        "name": "preprocess -> backbone",
        "logical": "image",
        "payload_fields": ["pre_to_tensor_bytes"],
    },
    {
        "name": "backbone -> encoder",
        "logical": "x=layer12, cache=[3,7]",
        "total_field": "model_backbone_to_encoder_bytes",
        "fallback_fields": ["model_p3_bytes", "model_p4_bytes", "model_p5_bytes"],
    },
    {
        "name": "encoder -> decoder",
        "logical": "cache=[21,24,27]",
        "total_field": "model_encoder_to_decoder_bytes",
    },
    {
        "name": "decoder -> A",
        "logical": "detections",
        "payload_fields": ["post_raw_decode_bytes"],
    },
]


def resolve_input_dir(input_dir: Path | None) -> Path:
    if input_dir is not None:
        path = input_dir.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input directory not found: {path}")
        return path
    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No default input directory found. Checked:\n{candidates}")


def model_name_from_csv(path: Path) -> str:
    match = re.match(r"local_original_(?:yolo|rtdetr)_(.+?)_FastSigmf_stagecost_", path.name)
    if match:
        return match.group(1)
    match = re.match(r"baseline_FastSigmf_(.+?)_local_[^_]+\.csv$", path.name)
    if match:
        return match.group(1)
    return path.stem


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_stage_means(csv_path: Path) -> tuple[dict[str, float], int]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    row_count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")
        numeric_fields = [
            field
            for field in reader.fieldnames
            if field.endswith("_sec")
            or field.endswith("_bytes")
            or field == "end_to_end_sec"
            or field
            in {
                "model_backbone_to_neck_bytes",
                "model_backbone_to_encoder_bytes",
                "model_encoder_to_decoder_bytes",
            }
        ]
        for row in reader:
            row_count += 1
            for field in numeric_fields:
                value = to_float(row.get(field))
                if value is None:
                    continue
                sums[field] = sums.get(field, 0.0) + value
                counts[field] = counts.get(field, 0) + 1
    means = {field: sums[field] / counts[field] for field in sums}
    return means, row_count


def module_boundary_specs(model_name: str) -> list[dict[str, Any]]:
    if "rtdetr" in model_name:
        return RTDETR_MODULE_BOUNDARIES
    if "yolo8" in model_name:
        return YOLO8_MODULE_BOUNDARIES
    if "yolo11" in model_name or "yolo26" in model_name:
        return YOLO11_26_MODULE_BOUNDARIES
    raise RuntimeError(f"Unsupported model for module-boundary payload plot: {model_name}")


def sum_stage_seconds(means: dict[str, float], stages: list[str]) -> float:
    return float(sum(float(means.get(f"{stage}_sec", 0.0) or 0.0) for stage in stages))


def collect_module_delay_values(model_name: str, means: dict[str, float]) -> tuple[list[str], np.ndarray]:
    modules: list[tuple[str, list[str]]] = [
        ("A preprocess", PREPROCESS_STAGES),
    ]
    if "rtdetr" in model_name:
        modules.extend(
            [
                ("B backbone", ["model_stem", "model_p2", "model_p3", "model_p4", "model_p5"]),
                ("C encoder", ["model_encoder"]),
                ("D decoder", ["model_decoder"]),
            ]
        )
    else:
        modules.extend(
            [
                ("B backbone", ["model_stem", "model_p2", "model_p3", "model_p4", "model_p5"]),
                ("C neck", ["model_neck"]),
                ("D head", ["model_head"]),
            ]
        )
    modules.append(("D postprocess", ["post_raw_decode"]))
    modules.append(
        (
            "A quality",
            ["post_load_label", "post_detection_metric", "post_overlap_metric"],
        )
    )

    names: list[str] = []
    delays: list[float] = []
    for name, stages in modules:
        delay = sum_stage_seconds(means, stages)
        if delay <= 0:
            continue
        names.append(name)
        delays.append(delay)
    return names, np.asarray(delays, dtype=float)


def boundary_payload_bytes(boundary: dict[str, Any], means: dict[str, float]) -> float:
    total_field = boundary.get("total_field")
    if total_field is not None:
        total = float(means.get(total_field, 0.0) or 0.0)
        if total > 0:
            return total

    fields = boundary.get("payload_fields") or boundary.get("fallback_fields") or []
    return sum(float(means.get(field, 0.0) or 0.0) for field in fields)


def collect_boundary_values(model_name: str, means: dict[str, float]) -> tuple[list[str], np.ndarray, list[str]]:
    names: list[str] = []
    payloads: list[float] = []
    logical: list[str] = []
    for boundary in module_boundary_specs(model_name):
        names.append(boundary["name"])
        payloads.append(boundary_payload_bytes(boundary, means))
        logical.append(boundary["logical"])
    return names, np.asarray(payloads, dtype=float), logical


def plot_model(
    model_name: str,
    csv_path: Path,
    output_dir: Path,
    rates_mbps: list[float],
) -> Path:
    means, row_count = read_stage_means(csv_path)
    module_names, module_delays = collect_module_delay_values(model_name, means)
    if not module_names:
        raise RuntimeError(f"No stage data found in {csv_path}")
    boundary_names, boundary_payload_bytes_array, boundary_logical = collect_boundary_values(model_name, means)

    payload_mb = boundary_payload_bytes_array / (1024.0 * 1024.0)
    transfer_sec = {
        rate: boundary_payload_bytes_array * 8.0 / (rate * 1_000_000.0)
        for rate in rates_mbps
    }

    y_module = np.arange(len(module_names))
    y_boundary = np.arange(len(boundary_names))
    row_count_for_height = max(len(module_names), len(boundary_names))
    fig_height = max(6.5, 0.65 * row_count_for_height + 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(23, fig_height))
    fig.suptitle(f"{model_name} local stage cost, samples={row_count}", fontsize=15)

    axes[0].barh(y_module, module_delays, color="#4c78a8")
    end_to_end_sec = means.get("end_to_end_sec", 0.0)
    if end_to_end_sec > 0:
        axes[0].axvline(
            end_to_end_sec,
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label=f"mean end-to-end {end_to_end_sec:.3f}s",
        )
        axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title("Average Module Delay")
    axes[0].set_xlabel("seconds")
    axes[0].set_yticks(y_module)
    axes[0].set_yticklabels(module_names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y_boundary, payload_mb, color="#72b7b2")
    axes[1].set_title("Minimal Module-Boundary Payload")
    axes[1].set_xlabel("MB")
    axes[1].set_yticks(y_boundary)
    axes[1].set_yticklabels(boundary_names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)
    for idx, detail in enumerate(boundary_logical):
        axes[1].text(
            payload_mb[idx] * 1.01,
            idx,
            detail,
            va="center",
            fontsize=8,
            color="#555555",
        )

    bar_height = 0.8 / max(1, len(rates_mbps))
    for idx, rate in enumerate(rates_mbps):
        offset = (idx - (len(rates_mbps) - 1) / 2.0) * bar_height
        label = f"{int(rate) if float(rate).is_integer() else rate} Mbps"
        axes[2].barh(y_boundary + offset, transfer_sec[rate], height=bar_height, label=label)
    if end_to_end_sec > 0:
        axes[2].axvline(
            end_to_end_sec,
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label=f"mean end-to-end {end_to_end_sec:.3f}s",
        )
    axes[2].set_title("Estimated Boundary Transfer Time")
    axes[2].set_xlabel("seconds")
    axes[2].set_yticks(y_boundary)
    axes[2].set_yticklabels(boundary_names, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the experiment parameters in new file names so plots for different sample limits, devices or thresholds do not overwrite each other.
    plot_stem = csv_path.stem if csv_path.name.startswith("local_original_") else safe_name(model_name)
    output_path = output_dir / f"{plot_stem}_stage_cost.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_rates(value: str) -> list[float]:
    rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rates:
        raise ValueError("--rates_mbps must contain at least one value")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot local FastSigmf stage-cost CSV results")
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--rates_mbps", default="25,50,150")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else input_dir / "stage_cost_plots"
    rates_mbps = parse_rates(args.rates_mbps)

    csv_paths = sorted(
        path for path in input_dir.glob("*.csv")
        if not path.name.endswith("summary.csv")
        and path.name.startswith(("baseline_FastSigmf_", "local_original_"))
    )
    if not csv_paths:
        raise FileNotFoundError(f"No local stage-cost detail CSV files found in {input_dir}")

    print(f"[plot] input_dir={input_dir}")
    print(f"[plot] output_dir={output_dir}")
    print(f"[plot] rates_mbps={rates_mbps}")
    for csv_path in csv_paths:
        model_name = model_name_from_csv(csv_path)
        output_path = plot_model(model_name, csv_path, output_dir, rates_mbps)
        print(f"[plot] {model_name}: {output_path}")


if __name__ == "__main__":
    main()
