"""
Plot distributed-stage-sweep results against local end-to-end baseline.

Defaults:
    local baseline:             ./local_test/local_baseline_results_FastSigmf_cpu
    distributed no-compress:    ./distributed_stage_sweep_results/distributed_stage_sweep_results_fullsamples
    distributed compressed 100: ./distributed_stage_sweep_results/distributed_stage_sweep_results_100samples_compress

One PNG is written per comparison metric. The x-axis is ordered by model family
so YOLO8, RT-DETR, YOLO11, and YOLO26 models stay grouped together.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTED_ROOT = PROJECT_ROOT / "distributed_stage_sweep_results"
DISTRIBUTED_FULLSAMPLES_NO_COMPRESS_DIR = DISTRIBUTED_ROOT / "distributed_stage_sweep_results_fullsamples"
DISTRIBUTED_100SAMPLES_COMPRESS_DIR = DISTRIBUTED_ROOT / "distributed_stage_sweep_results_100samples_compress"
BASELINE_DIR = PROJECT_ROOT / "local_test" / "local_baseline_results_FastSigmf_cpu"
OUTPUT_DIR = PROJECT_ROOT / "distributed_vs_baseline_plots"

MODEL_FAMILY_ORDER = ["yolo8", "rtdetr", "yolo11", "yolo26", "other"]
MODEL_SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
COMPRESSION_LABEL_ORDER = [
    "local baseline",
    "full no-compress",
    "nocompress",
    "quant_int8",
    "quant_int4",
    "topk_0p25",
    "topk_0p5",
    "topk_0p75",
    "llmint8_o0p01_fp16_int8",
]
DISTRIBUTED_COMPRESSOR_SOURCES = {"100sample compressed", "asym distributed compressor"}
DISTRIBUTED_SOURCES = {"fullsample no-compress", *DISTRIBUTED_COMPRESSOR_SOURCES}

METRICS = [
    ("inference_time_per_frame_s", "Inference time per frame", "s/frame"),
    ("mean_latency_s", "End-to-end latency", "s/sample"),
    ("overlap_accuracy", "Overlap Accuracy", "score"),
    ("overlap_precision", "Overlap Precision", "score"),
    ("overlap_recall", "Overlap Recall", "score"),
    ("overlap_f1", "Overlap F1", "score"),
    ("precision50", "Detection Precision@50", "score"),
    ("recall50", "Detection Recall@50", "score"),
    ("map50", "mAP50", "score"),
    ("map50_95", "mAP50-95", "score"),
]

STAGE_COMPONENTS = [
    ("avg_A_preprocess_sec", "A preprocess", "#4c78a8"),
    ("avg_B_compute_sec", "B compute", "#f58518"),
    ("avg_C_compute_sec", "C compute", "#54a24b"),
    ("avg_D_compute_sec", "D compute", "#b279a2"),
    ("avg_D_postprocess_sec", "D postprocess", "#b279a2"),
]

COMM_DELAY_COMPONENTS = [
    ("avg_a_to_b_send_sec", "A->B", "#4c78a8"),
    ("avg_b_to_c_send_sec", "B->C", "#f58518"),
    ("avg_c_to_d_send_sec", "C->D", "#54a24b"),
    ("avg_d_to_a_send_sec", "D->A", "#b279a2"),
]

COMM_PAYLOAD_COMPONENTS = [
    ("avg_a_to_b_payload_bytes", "A->B", "#4c78a8"),
    ("avg_b_to_c_payload_bytes", "B->C", "#f58518"),
    ("avg_c_to_d_payload_bytes", "C->D", "#54a24b"),
    ("avg_d_to_a_payload_bytes", "D->A", "#b279a2"),
]


@dataclass(frozen=True)
class ResultRow:
    model_name: str
    family_name: str
    method: str
    source: str
    compression_label: str
    rate_mbps: float | None
    sample_count: int | None
    metrics: dict[str, float]
    partition_id: str = ""
    partition_rank: int | None = None
    partition_cuts: str = ""
    detail_csv: str = ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "yolo8" in lowered or "yolo_v8" in lowered or "yolo-v8" in lowered:
        return "yolo8"
    if "rtdetr" in lowered:
        return "rtdetr"
    if "yolo11" in lowered:
        return "yolo11"
    if "yolo26" in lowered:
        return "yolo26"
    return "other"


def model_size_key(model_name: str) -> int:
    match = re.search(r"(?:yolo8|yolo[_-]?v8|yolo11|yolo26)([nsmxl])(?=_|[^a-z0-9]|$)", model_name, re.IGNORECASE)
    if match:
        return MODEL_SIZE_ORDER.get(match.group(1), 99)
    match = re.search(r"rtdetr_([nsmxl])\b", model_name)
    if match:
        return MODEL_SIZE_ORDER.get(match.group(1), 99)
    return 99


def model_sort_key(model_name: str) -> tuple[int, int, str]:
    family = model_family(model_name)
    return (
        MODEL_FAMILY_ORDER.index(family) if family in MODEL_FAMILY_ORDER else len(MODEL_FAMILY_ORDER),
        model_size_key(model_name),
        model_name,
    )


def short_model_label(model_name: str) -> str:
    match = re.match(r"sig_yolo_v8([nsmxl])_f(\d+)t(\d+)_c(\d+)$", model_name, re.IGNORECASE)
    if match:
        size, freq, time, classes = match.groups()
        return f"Asym-YOLOv8{size} f{freq}t{time} c{classes}"
    label = model_name
    label = label.removeprefix("cbrs_")
    label = label.replace("yolo", "YOLO")
    label = label.replace("rtdetr", "RT-DETR")
    label = label.replace("_", "-")
    return label


def latest_summary_csv(distributed_dir: Path) -> Path:
    candidates = sorted(
        distributed_dir.glob("*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No *_summary.csv found under {distributed_dir}")
    return candidates[0]


def normalize_partition_cuts(cuts: str) -> str:
    return cuts.strip().strip("[]").replace(",", "-").replace(" ", "")


def partition_label_from_fields(
    partition_id: str,
    partition_rank: int | None,
    partition_cuts: str,
) -> str:
    if partition_rank is not None:
        rank_label = f"rank{partition_rank:02d}"
    else:
        rank_match = re.search(r"rank(\d+)", partition_id, re.IGNORECASE)
        rank_label = f"rank{int(rank_match.group(1)):02d}" if rank_match else partition_id

    cuts_label = normalize_partition_cuts(partition_cuts)
    if not cuts_label and partition_id:
        cuts_match = re.search(r"cuts(.+)$", partition_id, re.IGNORECASE)
        if cuts_match:
            cuts_label = cuts_match.group(1).replace("_", "-")

    if rank_label and cuts_label:
        return f"{rank_label} cuts{cuts_label}"
    return rank_label or (f"cuts{cuts_label}" if cuts_label else "")


def partition_rank_label_from_fields(partition_id: str, partition_rank: int | None) -> str:
    if partition_rank is not None:
        return f"rank{partition_rank:02d}"
    rank_match = re.search(r"rank(\d+)", partition_id, re.IGNORECASE)
    if rank_match:
        return f"rank{int(rank_match.group(1)):02d}"
    return partition_id.split("_cuts", 1)[0]


def partition_label(row: ResultRow) -> str:
    return partition_label_from_fields(row.partition_id, row.partition_rank, row.partition_cuts)


def partition_rank_label(row: ResultRow) -> str:
    return partition_rank_label_from_fields(row.partition_id, row.partition_rank)


def load_distributed(
    summary_csv: Path,
    source: str,
    method_prefix: str,
    include_compression: bool,
    include_partitions: bool = False,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for row in read_csv_rows(summary_csv):
        model_name = row.get("model_name", "")
        rate_mbps = to_float(row.get("rate_mbps"))
        compression_label = row.get("compression_label", "").strip()
        partition_id = row.get("partition_id", "").strip() if include_partitions else ""
        partition_rank = to_optional_int(row.get("partition_rank")) if include_partitions else None
        partition_cuts = row.get("partition_cuts", "").strip() if include_partitions else ""
        partition_suffix = partition_label_from_fields(
            partition_id,
            partition_rank,
            partition_cuts,
        )
        if include_compression and compression_label:
            method = f"{method_prefix} {compression_label} {rate_mbps:g} Mbps"
        else:
            method = f"{method_prefix} {rate_mbps:g} Mbps"
        if partition_suffix:
            method = f"{method} {partition_suffix}"
        metrics = {
            "inference_time_per_frame_s": to_float(row.get("seconds_per_task")),
            "mean_latency_s": to_float(row.get("avg_end_to_end_sec"), default=np.nan),
            "seconds_per_task": to_float(row.get("seconds_per_task")),
            "avg_A_preprocess_sec": to_float(row.get("avg_A_preprocess_sec")),
            "avg_B_compute_sec": to_float(row.get("avg_B_compute_sec")),
            "avg_C_compute_sec": to_float(row.get("avg_C_compute_sec")),
            "avg_D_compute_sec": to_float(row.get("avg_D_compute_sec")),
            "avg_D_postprocess_sec": to_float(row.get("avg_D_postprocess_sec")),
            "avg_a_to_b_send_sec": to_float(row.get("avg_a_to_b_send_sec")),
            "avg_b_to_c_send_sec": to_float(row.get("avg_b_to_c_send_sec")),
            "avg_c_to_d_send_sec": to_float(row.get("avg_c_to_d_send_sec")),
            "avg_d_to_a_send_sec": to_float(row.get("avg_d_to_a_send_sec")),
            "avg_total_comm_sec": to_float(row.get("avg_total_comm_sec")),
            "avg_a_to_b_payload_bytes": to_float(row.get("avg_a_to_b_payload_bytes")),
            "avg_b_to_c_payload_bytes": to_float(row.get("avg_b_to_c_payload_bytes")),
            "avg_c_to_d_payload_bytes": to_float(row.get("avg_c_to_d_payload_bytes")),
            "avg_d_to_a_payload_bytes": to_float(row.get("avg_d_to_a_payload_bytes")),
            "avg_total_comm_bytes": to_float(row.get("avg_total_comm_bytes")),
            "overlap_accuracy": to_float(row.get("overlap_accuracy")),
            "overlap_precision": to_float(row.get("overlap_precision")),
            "overlap_recall": to_float(row.get("overlap_recall")),
            "overlap_f1": to_float(row.get("overlap_f1")),
            "precision50": to_float(row.get("precision50")),
            "recall50": to_float(row.get("recall50")),
            "map50": to_float(row.get("map50")),
            "map50_95": to_float(row.get("map50_95")),
        }
        rows.append(
            ResultRow(
                model_name=model_name,
                family_name=row.get("family_name") or model_family(model_name),
                method=method,
                source=source,
                compression_label=compression_label,
                rate_mbps=rate_mbps,
                sample_count=int(to_float(row.get("sample_count"), default=0.0)),
                metrics=metrics,
                partition_id=partition_id,
                partition_rank=partition_rank,
                partition_cuts=partition_cuts,
                detail_csv=row.get("detail_csv", "").strip(),
            )
        )
    return rows


def baseline_model_from_csv(path: Path) -> str:
    match = re.match(r"local_original_(?:yolo|rtdetr)_(.+?)_FastSigmf_stagecost_cpu_", path.name)
    if match:
        return match.group(1)
    prefix = "baseline_FastSigmf_"
    suffix = "_local_cpu.csv"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected baseline csv name: {path.name}")
    return name[len(prefix) : -len(suffix)]


def load_baseline_quality(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detection = data.get("detection", {})
    overlap = data.get("overlap", {})
    return {
        "overlap_accuracy": to_float(overlap.get("overlap_accuracy")),
        "overlap_precision": to_float(overlap.get("overlap_precision")),
        "overlap_recall": to_float(overlap.get("overlap_recall")),
        "overlap_f1": to_float(overlap.get("overlap_f1")),
        "precision50": to_float(detection.get("precision")),
        "recall50": to_float(detection.get("recall")),
        "map50": to_float(detection.get("map50")),
        "map50_95": to_float(detection.get("map50_95")),
    }


def load_baseline(baseline_dir: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    csv_paths = sorted({
        *baseline_dir.glob("baseline_FastSigmf_*_local_cpu.csv"),
        *(path for path in baseline_dir.glob("local_original_*_FastSigmf_stagecost_cpu_*.csv")
          if not path.name.endswith("summary.csv")),
    })
    seen_models: dict[str, Path] = {}
    for csv_path in csv_paths:
        model_name = baseline_model_from_csv(csv_path)
        csv_rows = read_csv_rows(csv_path)
        sample_count = len(csv_rows)
        if sample_count == 0:
            continue
        if model_name in seen_models:
            raise ValueError(
                f"Multiple local CPU runs for {model_name}: {seen_models[model_name].name}, {csv_path.name}. "
                "Use a baseline directory containing one experimental configuration per model."
            )
        seen_models[model_name] = csv_path

        completed_at = [to_float(row.get("completed_at_sec"), default=np.nan) for row in csv_rows]
        valid_completed = [v for v in completed_at if np.isfinite(v)]
        if valid_completed:
            total_sec = max(valid_completed)
        else:
            total_sec = sum(to_float(row.get("end_to_end_sec")) for row in csv_rows)

        metrics = {
            "inference_time_per_frame_s": total_sec / sample_count,
            "mean_latency_s": float(np.mean([to_float(r.get("end_to_end_sec"), default=np.nan) for r in csv_rows])),
        }

        quality_path = csv_path.with_name(csv_path.stem + "_quality.json")
        if quality_path.exists():
            metrics.update(load_baseline_quality(quality_path))
        else:
            metrics.update(
                {
                    "overlap_accuracy": 0.0,
                    "overlap_precision": 0.0,
                    "overlap_recall": 0.0,
                    "overlap_f1": 0.0,
                    "precision50": 0.0,
                    "recall50": 0.0,
                    "map50": 0.0,
                    "map50_95": 0.0,
                }
            )

        rows.append(
            ResultRow(
                model_name=model_name,
                family_name=model_family(model_name),
                method="local baseline",
                source="local baseline",
                compression_label="",
                rate_mbps=None,
                sample_count=sample_count,
                metrics=metrics,
            )
        )
    if not rows:
        raise FileNotFoundError(f"No local CPU baseline detail CSV found under {baseline_dir}")
    return rows


def method_sort_key(method: str) -> tuple[int, float, str]:
    if method in {"local baseline", "local cpu baseline"}:
        return (0, 0.0, method)
    if method == "nocompress" or " nocompress " in method:
        match = re.search(r"([0-9.]+) Mbps", method)
        rate = -float(match.group(1)) if match else 0.0
        return (1, rate, method)
    if method.startswith("full no-compress"):
        match = re.search(r"([0-9.]+) Mbps", method)
        rate = -float(match.group(1)) if match else 0.0
        return (1, rate, method)
    if method.startswith("compressed 100"):
        match = re.search(r"([0-9.]+) Mbps", method)
        rate = -float(match.group(1)) if match else 0.0
        return (2, rate, method)
    return (9, 0.0, method)


def method_base_label(method: str) -> str:
    if method in {"local baseline", "local cpu baseline"}:
        return "local baseline"
    if method.startswith("full no-compress"):
        return "full no-compress"
    for label in COMPRESSION_LABEL_ORDER:
        if method == label or method.startswith(f"{label} "):
            return label
    return method


def partition_rank_from_label(method: str) -> int:
    match = re.search(r"rank0*(\d+)", method, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def bandwidth_method_sort_key(method: str) -> tuple[int, int, str]:
    base = method_base_label(method)
    base_index = (
        COMPRESSION_LABEL_ORDER.index(base)
        if base in COMPRESSION_LABEL_ORDER
        else len(COMPRESSION_LABEL_ORDER)
    )
    return (base_index, partition_rank_from_label(method), method)


def flatten_rows(rows: list[ResultRow]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {
            "model_name": row.model_name,
            "family_name": row.family_name,
            "method": row.method,
            "source": row.source,
            "compression_label": row.compression_label,
            "rate_mbps": "" if row.rate_mbps is None else row.rate_mbps,
            "partition_id": row.partition_id,
            "partition_rank": "" if row.partition_rank is None else row.partition_rank,
            "partition_cuts": row.partition_cuts,
            "sample_count": "" if row.sample_count is None else row.sample_count,
            "detail_csv": row.detail_csv,
        }
        out.update(row.metrics)
        flat.append(out)
    return sorted(flat, key=lambda r: (model_sort_key(str(r["model_name"])), method_sort_key(str(r["method"]))))


def metric_matrix(rows: list[ResultRow], metric: str) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    models = sorted({row.model_name for row in rows}, key=model_sort_key)
    methods = sorted({row.method for row in rows}, key=method_sort_key)
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        values[(row.model_name, row.method)] = row.metrics.get(metric, np.nan)
    return models, methods, values


def rate_sort_key(rate: float) -> float:
    return -float(rate)


def rate_label(rate: float) -> str:
    value = float(rate)
    return str(int(value)) if value.is_integer() else str(value)


def method_label_for_bandwidth(row: ResultRow) -> str:
    if row.source == "local baseline":
        return "local baseline"
    if row.source == "fullsample no-compress":
        return "full no-compress"
    if row.source in DISTRIBUTED_COMPRESSOR_SOURCES:
        method_label = row.compression_label or "compressed"
        current_partition_rank_label = partition_rank_label(row)
        if current_partition_rank_label:
            return f"{method_label} {current_partition_rank_label}"
        return method_label
    return row.method


def short_method_label(method: str) -> str:
    base = method_base_label(method)
    rank = partition_rank_from_label(method)
    rank_suffix = f" p{rank}" if rank else ""
    if base == "local baseline":
        return "base"
    if base == "full no-compress":
        return "full"
    if base == "nocompress":
        return f"none{rank_suffix}"
    if base == "quant_int8":
        return f"q8{rank_suffix}"
    if base == "quant_int4":
        return f"q4{rank_suffix}"
    if base == "topk_0p25":
        return f"t25{rank_suffix}"
    if base == "topk_0p5":
        return f"t50{rank_suffix}"
    if base == "topk_0p75":
        return f"t75{rank_suffix}"
    if base == "llmint8_o0p01_fp16_int8":
        return f"llm{rank_suffix}"
    return method[:6]


def pipeline_bottleneck_label(row: ResultRow) -> tuple[str, bool] | None:
    candidates: list[tuple[str, bool, float]] = [
        ("A-pre", True, row.metrics.get("avg_A_preprocess_sec", np.nan)),
        ("B-comp", True, row.metrics.get("avg_B_compute_sec", np.nan)),
        ("C-comp", True, row.metrics.get("avg_C_compute_sec", np.nan)),
        ("D-comp", True, row.metrics.get("avg_D_compute_sec", np.nan)),
        ("D-post", True, row.metrics.get("avg_D_postprocess_sec", np.nan)),
        ("A>B", False, row.metrics.get("avg_a_to_b_send_sec", np.nan)),
        ("B>C", False, row.metrics.get("avg_b_to_c_send_sec", np.nan)),
        ("C>D", False, row.metrics.get("avg_c_to_d_send_sec", np.nan)),
        ("D>A", False, row.metrics.get("avg_d_to_a_send_sec", np.nan)),
    ]
    finite_candidates = [
        (label, is_compute, float(value))
        for label, is_compute, value in candidates
        if np.isfinite(value)
    ]
    if not finite_candidates:
        return None
    label, is_compute, _value = max(finite_candidates, key=lambda item: item[2])
    return label, is_compute


def bandwidth_methods(rows: list[ResultRow]) -> list[str]:
    labels = {method_label_for_bandwidth(row) for row in rows}
    return sorted(labels, key=bandwidth_method_sort_key)


def bandwidth_metric_values(
    rows: list[ResultRow],
    metric: str,
) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    models = sorted({row.model_name for row in rows}, key=model_sort_key)
    methods = bandwidth_methods(rows)
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        values[(row.model_name, method_label_for_bandwidth(row))] = row.metrics.get(metric, np.nan)
    return models, methods, values


def distributed_rows_only(rows: list[ResultRow]) -> list[ResultRow]:
    return [row for row in rows if row.source in DISTRIBUTED_SOURCES]


def component_value(row: ResultRow, field: str, scale: float = 1.0) -> float:
    return row.metrics.get(field, np.nan) / scale


def e2e_components(row: ResultRow) -> list[tuple[str, str, float]]:
    stage_total = sum(row.metrics.get(field, 0.0) for field, _label, _color in STAGE_COMPONENTS)
    comm_total = row.metrics.get("avg_total_comm_sec", 0.0)
    e2e = row.metrics.get("inference_time_per_frame_s", row.metrics.get("seconds_per_task", 0.0))
    residual = max(float(e2e) - float(stage_total) - float(comm_total), 0.0)
    return [
        *[(label, color, row.metrics.get(field, np.nan)) for field, label, color in STAGE_COMPONENTS],
        ("communication", "#72b7b2", comm_total),
        ("residual", "#bab0ac", residual),
    ]


def component_list_values(
    row: ResultRow,
    components: list[tuple[str, str, str]],
    scale: float = 1.0,
) -> list[tuple[str, str, float]]:
    return [(label, color, component_value(row, field, scale=scale)) for field, label, color in components]


def add_family_groups(ax: plt.Axes, models: list[str]) -> None:
    if not models:
        return
    families = [model_family(model) for model in models]
    start = 0
    for i in range(1, len(models) + 1):
        if i == len(models) or families[i] != families[start]:
            end = i - 1
            if start > 0:
                ax.axvline(start - 0.5, color="#cccccc", linewidth=0.8)
            center = (start + end) / 2
            ax.text(
                center,
                -0.23,
                families[start].upper(),
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                color="#444444",
            )
            start = i


def plot_metric(rows: list[ResultRow], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    models, methods, values = metric_matrix(rows, metric)
    x = np.arange(len(models), dtype=float)
    width = min(0.86 / max(len(methods), 1), 0.16)

    fig_width = max(14.0, len(models) * max(1.25, len(methods) * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    colors = [
        "#3b6ea8",
        "#d95f02",
        "#1b9e77",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#1f78b4",
        "#b2df8a",
        "#fb9a99",
        "#cab2d6",
    ]

    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        y = [values.get((model, method), np.nan) for model in models]
        ax.bar(
            x + offset,
            y,
            width=width,
            label=method,
            color=colors[idx % len(colors)],
            edgecolor="#222222",
            linewidth=0.35,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([short_model_label(model) for model in models], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncols=min(len(methods), 4), fontsize=8)
    add_family_groups(ax, models)

    if metric != "inference_time_per_frame_s":
        ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metric_by_bandwidth(rows: list[ResultRow], metric: str, title: str, ylabel: str, output_path: Path) -> None:
    rates = sorted(
        {float(row.rate_mbps) for row in rows if row.rate_mbps is not None},
        key=rate_sort_key,
    )
    if not rates:
        return

    baseline_rows = [row for row in rows if row.source == "local baseline"]
    all_models = sorted({row.model_name for row in rows}, key=model_sort_key)
    x = np.arange(len(all_models), dtype=float)
    methods = bandwidth_methods(rows)
    width = min(0.86 / max(len(methods), 1), 0.12)
    colors = {
        "local baseline": "#3b6ea8",
        "full no-compress": "#d95f02",
        "nocompress": "#d95f02",
        "quant_int8": "#1b9e77",
        "quant_int4": "#66a61e",
        "topk_0p25": "#7570b3",
        "topk_0p5": "#e7298a",
        "topk_0p75": "#e6ab02",
        "llmint8_o0p01_fp16_int8": "#a6761d",
    }

    fig_width = max(15.0, len(all_models) * 1.45)
    fig_height = max(4.2 * len(rates), 7.0)
    fig, axes = plt.subplots(
        nrows=len(rates),
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=False,
    )
    if len(rates) == 1:
        axes = [axes]

    handles = []
    labels = []
    for ax, rate in zip(axes, rates):
        rate_rows = baseline_rows + [
            row
            for row in rows
            if row.rate_mbps is not None and float(row.rate_mbps) == float(rate)
        ]
        row_map = {
            (row.model_name, method_label_for_bandwidth(row)): row
            for row in rate_rows
        }
        _models, rate_methods, values = bandwidth_metric_values(rate_rows, metric)
        finite_metric_values = [
            value
            for model in all_models
            for method in rate_methods
            for value in [values.get((model, method), np.nan)]
            if np.isfinite(value)
        ]
        label_offset = (max(finite_metric_values) * 0.012) if finite_metric_values else 0.01
        for idx, method in enumerate(rate_methods):
            offset = (idx - (len(rate_methods) - 1) / 2) * width
            y = [values.get((model, method), np.nan) for model in all_models]
            bars = ax.bar(
                x + offset,
                y,
                width=width,
                label=method,
                color=colors.get(method_base_label(method), "#666666"),
                edgecolor="#222222",
                linewidth=0.3,
            )
            if method not in labels:
                handles.append(bars[0])
                labels.append(method)
            for model, xpos, value, bar in zip(all_models, x + offset, y, bars):
                if not np.isfinite(value):
                    continue
                label_text = short_method_label(method)
                bbox = None
                row = row_map.get((model, method))
                if row is not None and row.source != "local baseline":
                    bottleneck = pipeline_bottleneck_label(row)
                    if bottleneck is not None:
                        bottleneck_label, is_compute_bottleneck = bottleneck
                        label_text = f"{label_text}\n{bottleneck_label}"
                        bbox = {
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "#ffffff",
                            "edgecolor": "#555555",
                            "linewidth": 0.25,
                            "alpha": 0.86,
                        }
                        if is_compute_bottleneck:
                            bar.set_hatch("///")
                            bar.set_linewidth(0.45)
                ax.text(
                    xpos,
                    value + label_offset,
                    label_text,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=5.6,
                    color="#222222",
                    bbox=bbox,
                )

        ax.set_title(f"{rate_label(rate)} Mbps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([short_model_label(model) for model in all_models], rotation=35, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        add_family_groups(ax, all_models)
        if finite_metric_values:
            current_top = max(finite_metric_values)
            if current_top > 0:
                ax.set_ylim(top=current_top * 1.28)
        if metric != "inference_time_per_frame_s":
            ax.set_ylim(0.0, max(ax.get_ylim()[1], 1.12))

    fig.suptitle(title, y=0.995)
    fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 4), fontsize=8, bbox_to_anchor=(0.5, 0.965))
    fig.text(
        0.5,
        0.915,
        "Bottleneck marker: hatched bar = compute-stage bottleneck; solid bar = communication-stage bottleneck",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_stacked_components_by_bandwidth(
    rows: list[ResultRow],
    title: str,
    ylabel: str,
    output_path: Path,
    components_for_row,
) -> None:
    rows = distributed_rows_only(rows)
    rates = sorted(
        {float(row.rate_mbps) for row in rows if row.rate_mbps is not None},
        key=rate_sort_key,
    )
    if not rates:
        return

    all_models = sorted({row.model_name for row in rows}, key=model_sort_key)
    x = np.arange(len(all_models), dtype=float)
    methods = bandwidth_methods(rows)
    width = min(0.86 / max(len(methods), 1), 0.12)

    fig_width = max(15.0, len(all_models) * 1.45)
    fig_height = max(4.4 * len(rates), 7.0)
    fig, axes = plt.subplots(
        nrows=len(rates),
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=False,
    )
    if len(rates) == 1:
        axes = [axes]

    legend_seen: dict[str, Any] = {}
    for ax, rate in zip(axes, rates):
        rate_rows = [
            row
            for row in rows
            if row.rate_mbps is not None and float(row.rate_mbps) == float(rate)
        ]
        row_map = {
            (row.model_name, method_label_for_bandwidth(row)): row
            for row in rate_rows
        }
        for method_idx, method in enumerate(methods):
            offset = (method_idx - (len(methods) - 1) / 2) * width
            bottoms = np.zeros(len(all_models), dtype=float)
            method_rows = [row_map.get((model, method)) for model in all_models]
            component_names: list[str] = []
            for row in method_rows:
                if row is not None:
                    component_names = [label for label, _color, _value in components_for_row(row)]
                    break
            for component_name in component_names:
                y_values: list[float] = []
                color = "#666666"
                for row in method_rows:
                    if row is None:
                        y_values.append(np.nan)
                        continue
                    components = {label: (color_value, value) for label, color_value, value in components_for_row(row)}
                    color, value = components[component_name]
                    y_values.append(value)
                bars = ax.bar(
                    x + offset,
                    y_values,
                    bottom=bottoms,
                    width=width,
                    label=component_name,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.25,
                )
                clean_values = np.array([0.0 if not np.isfinite(value) else value for value in y_values])
                bottoms += clean_values
                if component_name not in legend_seen:
                    legend_seen[component_name] = bars[0]

        ax.set_title(f"{rate_label(rate)} Mbps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([short_model_label(model) for model in all_models], rotation=35, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        add_family_groups(ax, all_models)

    fig.suptitle(title, y=0.995)
    fig.legend(
        list(legend_seen.values()),
        list(legend_seen.keys()),
        loc="upper center",
        ncols=min(len(legend_seen), 4),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def bottleneck(values: list[tuple[str, float]]) -> tuple[str, float]:
    finite = [(label, value) for label, value in values if np.isfinite(value)]
    if not finite:
        return "", np.nan
    return max(finite, key=lambda item: item[1])


def plot_bottlenecks_by_bandwidth(rows: list[ResultRow], output_path: Path) -> None:
    rows = distributed_rows_only(rows)
    rates = sorted(
        {float(row.rate_mbps) for row in rows if row.rate_mbps is not None},
        key=rate_sort_key,
    )
    if not rates:
        return

    all_models = sorted({row.model_name for row in rows}, key=model_sort_key)
    x = np.arange(len(all_models), dtype=float)
    methods = bandwidth_methods(rows)
    width = min(0.86 / max(len(methods), 1), 0.12)
    method_colors = {
        "full no-compress": "#d95f02",
        "nocompress": "#d95f02",
        "quant_int8": "#1b9e77",
        "quant_int4": "#66a61e",
        "topk_0p25": "#7570b3",
        "topk_0p5": "#e7298a",
        "topk_0p75": "#e6ab02",
        "llmint8_o0p01_fp16_int8": "#a6761d",
    }

    fig_width = max(15.0, len(all_models) * 1.45)
    fig_height = max(6.5 * len(rates), 9.0)
    fig, axes = plt.subplots(
        nrows=len(rates),
        ncols=2,
        figsize=(fig_width, fig_height),
        sharex=False,
    )
    if len(rates) == 1:
        axes = np.array([axes])

    legend_seen: dict[str, Any] = {}
    for rate_idx, rate in enumerate(rates):
        rate_rows = [
            row
            for row in rows
            if row.rate_mbps is not None and float(row.rate_mbps) == float(rate)
        ]
        row_map = {
            (row.model_name, method_label_for_bandwidth(row)): row
            for row in rate_rows
        }
        compute_ax = axes[rate_idx][0]
        comm_ax = axes[rate_idx][1]
        for method_idx, method in enumerate(methods):
            offset = (method_idx - (len(methods) - 1) / 2) * width
            compute_heights = []
            compute_labels = []
            comm_heights = []
            comm_labels = []
            for model in all_models:
                row = row_map.get((model, method))
                if row is None:
                    compute_heights.append(np.nan)
                    compute_labels.append("")
                    comm_heights.append(np.nan)
                    comm_labels.append("")
                    continue
                compute_label, compute_value = bottleneck(
                    [(label, row.metrics.get(field, np.nan)) for field, label, _color in STAGE_COMPONENTS]
                )
                comm_label, comm_value = bottleneck(
                    [(label, row.metrics.get(field, np.nan)) for field, label, _color in COMM_DELAY_COMPONENTS]
                )
                compute_heights.append(compute_value)
                compute_labels.append(compute_label.replace(" compute", "").replace(" preprocess", ""))
                comm_heights.append(comm_value)
                comm_labels.append(comm_label)
            color = method_colors.get(method_base_label(method), "#666666")
            compute_bars = compute_ax.bar(
                x + offset,
                compute_heights,
                width=width,
                color=color,
                edgecolor="#222222",
                linewidth=0.25,
                label=method,
            )
            comm_ax.bar(
                x + offset,
                comm_heights,
                width=width,
                color=color,
                edgecolor="#222222",
                linewidth=0.25,
                label=method,
            )
            if method not in legend_seen:
                legend_seen[method] = compute_bars[0]

            for ax, heights, text_labels in (
                (compute_ax, compute_heights, compute_labels),
                (comm_ax, comm_heights, comm_labels),
            ):
                finite_values = [value for value in heights if np.isfinite(value)]
                max_height = max(finite_values) if finite_values else 0.0
                if max_height <= 0:
                    continue
                for xpos, height, text in zip(x + offset, heights, text_labels):
                    if not np.isfinite(height) or not text:
                        continue
                    ax.text(
                        xpos,
                        height + max_height * 0.015,
                        text,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6,
                    )

        compute_ax.set_title(f"{rate_label(rate)} Mbps: max compute stage")
        comm_ax.set_title(f"{rate_label(rate)} Mbps: max communication link")
        for ax in (compute_ax, comm_ax):
            ax.set_ylabel("sec / sample")
            ax.set_xticks(x)
            ax.set_xticklabels([short_model_label(model) for model in all_models], rotation=35, ha="right")
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            add_family_groups(ax, all_models)

    fig.suptitle("Bottleneck Stage and Communication Link", y=0.995)
    fig.legend(
        list(legend_seen.values()),
        list(legend_seen.keys()),
        loc="upper center",
        ncols=min(len(legend_seen), 4),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def stage_comm_components(row: ResultRow) -> list[tuple[str, str, float]]:
    return [
        *[(label, color, row.metrics.get(field, np.nan)) for field, label, color in STAGE_COMPONENTS],
        *[(f"comm {label}", color, row.metrics.get(field, np.nan)) for field, label, color in COMM_DELAY_COMPONENTS],
    ]


def stage_comm_methods_for_rate(rows: list[ResultRow], rate: float) -> list[str]:
    rate_rows = [
        row
        for row in rows
        if row.rate_mbps is not None and float(row.rate_mbps) == float(rate)
    ]
    labels = {method_label_for_bandwidth(row) for row in rate_rows}
    return sorted(labels, key=bandwidth_method_sort_key)


def plot_stage_comm_for_rate(
    rows: list[ResultRow],
    baseline_rows: list[ResultRow],
    rate: float,
    output_path: Path,
) -> None:
    rate_rows = [
        row
        for row in rows
        if row.rate_mbps is not None and float(row.rate_mbps) == float(rate)
    ]
    if not rate_rows:
        return

    models = sorted({row.model_name for row in rate_rows}, key=model_sort_key)
    methods = stage_comm_methods_for_rate(rows, rate)
    row_map = {
        (row.model_name, method_label_for_bandwidth(row)): row
        for row in rate_rows
    }
    baseline_delay = {
        row.model_name: row.metrics.get("inference_time_per_frame_s", np.nan)
        for row in baseline_rows
    }

    component_order = [
        "A preprocess",
        "B compute",
        "C compute",
        "D compute",
        "D postprocess",
        "comm A->B",
        "comm B->C",
        "comm C->D",
        "comm D->A",
    ]
    component_colors = {
        "A preprocess": "#4c78a8",
        "B compute": "#f58518",
        "C compute": "#54a24b",
        "D compute": "#b279a2",
        "D postprocess": "#b279a2",
        "comm A->B": "#72b7b2",
        "comm B->C": "#ff9da6",
        "comm C->D": "#9d755d",
        "comm D->A": "#bab0ac",
    }
    component_hatches = {
        component: "///" if not component.startswith("comm ") else ""
        for component in component_order
    }

    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig_width = max(15.5, len(models) * 1.35)
    fig_height = max(4.1 * nrows, 7.2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=False, sharey=True)
    axes_array = np.array(axes).reshape(-1)
    x = np.arange(len(models), dtype=float)
    legend_seen: dict[str, Any] = {}
    line_seen: dict[str, Any] = {}

    y_max = 0.0
    for row in rate_rows:
        y_max = max(y_max, row.metrics.get("inference_time_per_frame_s", 0.0))
        y_max = max(y_max, sum(value for _label, _color, value in stage_comm_components(row) if np.isfinite(value)))
    for value in baseline_delay.values():
        if np.isfinite(value):
            y_max = max(y_max, float(value))
    y_max = y_max * 1.18 if y_max > 0 else 1.0

    for ax_idx, method in enumerate(methods):
        ax = axes_array[ax_idx]
        bottoms = np.zeros(len(models), dtype=float)
        for component in component_order:
            values = []
            for model in models:
                row = row_map.get((model, method))
                if row is None:
                    values.append(np.nan)
                    continue
                value_by_component = {label: value for label, _color, value in stage_comm_components(row)}
                values.append(value_by_component.get(component, np.nan))
            bars = ax.bar(
                x,
                values,
                bottom=bottoms,
                width=0.72,
                color=component_colors[component],
                hatch=component_hatches[component],
                edgecolor="#222222",
                linewidth=0.35,
                label=component,
            )
            clean_values = np.array([0.0 if not np.isfinite(value) else value for value in values])
            bottoms += clean_values
            if component not in legend_seen:
                legend_seen[component] = bars[0]

        distributed_e2e = [
            row_map[(model, method)].metrics.get("inference_time_per_frame_s", np.nan)
            if (model, method) in row_map
            else np.nan
            for model in models
        ]
        baseline_e2e = [baseline_delay.get(model, np.nan) for model in models]
        dist_line = ax.plot(
            x,
            distributed_e2e,
            color="#111111",
            marker="o",
            linewidth=1.4,
            markersize=3.5,
            label="distributed avg e2e",
        )[0]
        base_line = ax.plot(
            x,
            baseline_e2e,
            color="#d62728",
            marker="x",
            linestyle="--",
            linewidth=1.2,
            markersize=4.0,
            label="local baseline avg e2e",
        )[0]
        line_seen.setdefault("distributed avg e2e", dist_line)
        line_seen.setdefault("local baseline avg e2e", base_line)

        ax.set_title(method)
        ax.set_ylabel("sec / sample")
        ax.set_ylim(0.0, y_max)
        ax.set_xticks(x)
        ax.set_xticklabels([short_model_label(model) for model in models], rotation=35, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        add_family_groups(ax, models)

    for unused_ax in axes_array[len(methods):]:
        unused_ax.axis("off")

    handles = list(legend_seen.values()) + list(line_seen.values())
    labels = list(legend_seen.keys()) + list(line_seen.keys())
    fig.suptitle(
        f"Stage and Communication Breakdown with E2E Delay Lines: {rate_label(rate)} Mbps",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.975),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_stage_comm_plots(rows: list[ResultRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    distributed_rows = distributed_rows_only(rows)
    baseline_rows = [row for row in rows if row.source == "local baseline"]
    rates = sorted(
        {float(row.rate_mbps) for row in distributed_rows if row.rate_mbps is not None},
        key=rate_sort_key,
    )
    for rate in rates:
        plot_stage_comm_for_rate(
            rows=distributed_rows,
            baseline_rows=baseline_rows,
            rate=rate,
            output_path=output_dir / f"stage_comm_by_bandwidth_{rate_label(rate)}mbps.png",
        )


def write_bandwidth_subset_csvs(rows: list[ResultRow], output_dir: Path) -> None:
    by_bandwidth_dir = output_dir / "by_bandwidth"
    for rate in sorted({float(row.rate_mbps) for row in rows if row.rate_mbps is not None}, key=rate_sort_key):
        subset = [
            row
            for row in rows
            if row.source == "local baseline"
            or (row.rate_mbps is not None and float(row.rate_mbps) == float(rate))
        ]
        write_csv_rows(
            by_bandwidth_dir / f"comparison_{rate_label(rate)}mbps.csv",
            flatten_rows(subset),
        )


def build_plots(
    baseline_dir: Path,
    fullsamples_no_compress_summary: Path,
    compressed_100_summary: Path,
    output_dir: Path,
) -> None:
    baseline_rows = load_baseline(baseline_dir)
    full_no_compress_rows = load_distributed(
        fullsamples_no_compress_summary,
        source="fullsample no-compress",
        method_prefix="full no-compress",
        include_compression=False,
    )
    compressed_100_rows = load_distributed(
        compressed_100_summary,
        source="100sample compressed",
        method_prefix="compressed 100",
        include_compression=True,
    )
    all_rows = baseline_rows + full_no_compress_rows + compressed_100_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "distributed_vs_baseline_comparison.csv", flatten_rows(all_rows))

    by_bandwidth_dir = output_dir / "by_bandwidth"
    by_bandwidth_dir.mkdir(parents=True, exist_ok=True)
    write_bandwidth_subset_csvs(all_rows, output_dir)
    for metric, title, ylabel in METRICS:
        plot_metric_by_bandwidth(
            all_rows,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output_path=by_bandwidth_dir / f"{metric}.png",
        )
    build_stage_comm_plots(all_rows, output_dir / "stage_comm")

    print(f"[plot] baseline dir: {baseline_dir}")
    print(f"[plot] fullsample no-compress summary: {fullsamples_no_compress_summary}")
    print(f"[plot] 100sample compressed summary: {compressed_100_summary}")
    print(f"[plot] comparison csv: {output_dir / 'distributed_vs_baseline_comparison.csv'}")
    print(f"[plot] bandwidth-grouped figures written to: {by_bandwidth_dir}")
    print(f"[plot] stage/communication figures written to: {output_dir / 'stage_comm'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot distributed vs local baseline metrics.")
    parser.add_argument("--baseline_dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--fullsamples_no_compress_summary", type=Path, default=None)
    parser.add_argument("--fullsamples_no_compress_dir", type=Path, default=DISTRIBUTED_FULLSAMPLES_NO_COMPRESS_DIR)
    parser.add_argument("--compressed_100_summary", type=Path, default=None)
    parser.add_argument("--compressed_100_dir", type=Path, default=DISTRIBUTED_100SAMPLES_COMPRESS_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fullsamples_no_compress_summary = (
        args.fullsamples_no_compress_summary
        or latest_summary_csv(args.fullsamples_no_compress_dir)
    )
    compressed_100_summary = args.compressed_100_summary or latest_summary_csv(args.compressed_100_dir)
    build_plots(
        baseline_dir=args.baseline_dir,
        fullsamples_no_compress_summary=fullsamples_no_compress_summary,
        compressed_100_summary=compressed_100_summary,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
