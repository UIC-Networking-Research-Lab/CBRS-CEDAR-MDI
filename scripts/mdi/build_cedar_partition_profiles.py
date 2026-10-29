from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import csv
import argparse
import ast
import itertools
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All configuration is intentionally global. This script is a reproducible
# offline analysis pass, not a command-line experiment runner.
LOCAL_BASELINE_DIR = PROJECT_ROOT / "local_baseline_results_by_stage_cost_asym_yolo"
DISTRIBUTED_RESULTS_DIR = PROJECT_ROOT / "distributed_stage_sweep_results_asym"
OUTPUT_DIR = PROJECT_ROOT / "experiment_reruns" / "offline_profiles"

PLATFORM = "jetson"
DEVICE = "cpu"
MODEL_NAMES = (
    "sig_yolo_v8n_f20t20_c100",
    "sig_yolo_v8s_f20t20_c100",
    "sig_yolo_v8x_f20t20_c100",
)
RATE_GROUPS_MBPS = (25.0, 50.0, 150.0)
NODES = ("A", "B", "C", "D")
LINKS = ("A_to_B", "B_to_C", "C_to_D", "D_to_A")
CHAIN_LINKS = ("A_to_B", "B_to_C", "C_to_D")

TOP_K = 16
STAGE_STAT = "median"
BYTE_STAT = "median"
BANDWIDTH_STAT = "median"

FORBID_PREPROCESS_INTERNAL_COMPRESSION = True
ALLOW_UNSUPPORTED_RUNTIME_BOUNDARIES = False
PRINT_INPUT_AUDIT = True
PRINT_FOLDER_FILE_LIMIT = 20
PRINT_READ_FILE_LIMIT = 100

READ_AUDIT: list[dict[str, Any]] = []


def f(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    variance = statistics.pvariance(clean) if len(clean) > 1 else 0.0
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "variance": variance,
        "std": math.sqrt(variance),
        "p90": percentile(clean, 90.0),
        "p95": percentile(clean, 95.0),
    }


def stat_value(stat: dict[str, Any], name: str) -> float:
    if name not in stat:
        raise KeyError(f"Unknown statistic {name!r}; available={sorted(stat)}")
    return float(stat[name])


def format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def csv_directory_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "csv_count": 0,
            "total_csv_bytes": 0,
            "latest_csv_mtime": None,
            "files": [],
        }
    files = sorted(path.glob("*.csv"))
    file_rows: list[dict[str, Any]] = []
    for file_path in files:
        stat = file_path.stat()
        file_rows.append(
            {
                "name": file_path.name,
                "path": str(file_path),
                "bytes": stat.st_size,
                "modified": format_mtime(stat.st_mtime),
                "mtime": stat.st_mtime,
            }
        )
    latest_mtime = max((item["mtime"] for item in file_rows), default=None)
    return {
        "path": str(path),
        "exists": True,
        "csv_count": len(file_rows),
        "total_csv_bytes": sum(item["bytes"] for item in file_rows),
        "latest_csv_mtime": format_mtime(latest_mtime) if latest_mtime else None,
        "files": sorted(file_rows, key=lambda item: item["mtime"], reverse=True),
    }


def print_input_folder_snapshots() -> None:
    if not PRINT_INPUT_AUDIT:
        return
    print("[input-folders]")
    for label, path in (
        ("local_baseline", LOCAL_BASELINE_DIR),
        ("distributed_results", DISTRIBUTED_RESULTS_DIR),
        ("output", OUTPUT_DIR),
    ):
        snapshot = csv_directory_snapshot(path)
        print(
            f"  {label}: path={snapshot['path']} exists={snapshot['exists']} "
            f"csv_count={snapshot['csv_count']} total_csv_bytes={snapshot['total_csv_bytes']} "
            f"latest_csv_mtime={snapshot['latest_csv_mtime']}"
        )
        for item in snapshot["files"][:PRINT_FOLDER_FILE_LIMIT]:
            print(f"    - {item['name']} bytes={item['bytes']} modified={item['modified']}")
        remaining = max(0, snapshot["csv_count"] - PRINT_FOLDER_FILE_LIMIT)
        if remaining:
            print(f"    ... {remaining} more csv files")


def print_read_audit() -> None:
    if not PRINT_INPUT_AUDIT:
        return
    total_elapsed = sum(float(item["read_elapsed_sec"]) for item in READ_AUDIT)
    total_rows = sum(int(item["rows"]) for item in READ_AUDIT)
    total_bytes = sum(int(item["bytes"]) for item in READ_AUDIT)
    print(
        f"[read-audit] files={len(READ_AUDIT)} rows={total_rows} "
        f"bytes={total_bytes} total_read_sec={total_elapsed:.6f}"
    )
    by_dir: dict[str, dict[str, float]] = defaultdict(lambda: {"files": 0, "rows": 0, "bytes": 0, "read_sec": 0.0})
    for item in READ_AUDIT:
        directory = str(Path(item["path"]).parent)
        by_dir[directory]["files"] += 1
        by_dir[directory]["rows"] += int(item["rows"])
        by_dir[directory]["bytes"] += int(item["bytes"])
        by_dir[directory]["read_sec"] += float(item["read_elapsed_sec"])
    for directory, values in sorted(by_dir.items()):
        print(
            f"  dir={directory} files={int(values['files'])} rows={int(values['rows'])} "
            f"bytes={int(values['bytes'])} read_sec={values['read_sec']:.6f}"
        )
    for item in READ_AUDIT[:PRINT_READ_FILE_LIMIT]:
        print(
            f"    - {Path(item['path']).name} rows={item['rows']} bytes={item['bytes']} "
            f"modified={item['modified']} read_sec={float(item['read_elapsed_sec']):.6f}"
        )
    remaining = max(0, len(READ_AUDIT) - PRINT_READ_FILE_LIMIT)
    if remaining:
        print(f"    ... {remaining} more read files")


def read_csv(path: Path) -> list[dict[str, str]]:
    start = time.perf_counter()
    stat = path.stat()
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    READ_AUDIT.append(
        {
            "path": str(path),
            "rows": len(rows),
            "bytes": stat.st_size,
            "modified": format_mtime(stat.st_mtime),
            "read_elapsed_sec": time.perf_counter() - start,
        }
    )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True)
class StageDef:
    name: str
    compute_fields: tuple[str, ...]
    output_byte_fields: tuple[str, ...]
    payload_type: str
    runtime_boundary_supported: bool
    preprocess_internal_boundary: bool
    compression_allowed: bool


STAGES: tuple[StageDef, ...] = (
    StageDef("pre_read_iq", ("pre_read_iq_sec",), ("pre_read_iq_bytes",), "iq_complex64", False, True, False),
    StageDef("pre_spectrogram", ("pre_spectrogram_sec",), ("pre_spectrogram_bytes",), "spectrogram_magnitude", False, True, False),
    StageDef("pre_fftshift_log", ("pre_fftshift_log_sec",), ("pre_fftshift_log_bytes",), "spectrogram_db", True, True, False),
    StageDef(
        "pre_percentile_clip",
        ("pre_percentile_clip_sec",),
        ("pre_fftshift_log_bytes", "pre_percentile_clip_bytes"),
        "spectrogram_db_with_clip_range",
        True,
        True,
        False,
    ),
    StageDef("pre_canvas_render", ("pre_canvas_render_sec",), ("pre_canvas_render_bytes",), "rgba_render_buffer", True, True, False),
    StageDef("pre_canvas_to_rgb", ("pre_canvas_to_rgb_sec",), ("pre_canvas_to_rgb_bytes",), "rgb_image", True, True, False),
    StageDef("pre_to_tensor", ("pre_to_tensor_sec",), ("pre_to_tensor_bytes",), "image_tensor", True, False, True),
    StageDef("model_stage_0", ("model_stage_0_sec",), ("model_stage_0_bytes",), "model_stage_0_output", True, False, True),
    StageDef("model_stage_1", ("model_stage_1_sec",), ("model_stage_1_bytes",), "model_stage_1_output", True, False, True),
    StageDef("model_stage_2", ("model_stage_2_sec",), ("model_stage_2_bytes",), "model_stage_2_output", True, False, True),
    StageDef("model_stage_3", ("model_stage_3_sec",), ("model_stage_3_bytes",), "model_stage_3_output", True, False, True),
    StageDef("model_stage_4", ("model_stage_4_sec",), ("model_stage_4_bytes",), "model_stage_4_output", True, False, True),
    StageDef("model_sppf", ("model_sppf_sec",), ("model_sppf_bytes",), "model_sppf_output", True, False, True),
    # Keep PSA/head/postprocess as an atomic runtime tail. The AsymYOLO head is
    # not a linear layer boundary for distributed placement.
    StageDef("model_psa", ("model_psa_sec",), ("model_psa_bytes",), "model_psa_output", False, False, False),
    # The runtime's decoded_prediction payload is the raw_output tuple:
    # y + box_raw + cls_raw. Use all three local byte fields, not y alone.
    StageDef(
        "model_head",
        ("model_head_box_sec", "model_head_cls_sec", "model_head_decode_sec"),
        ("model_head_decode_bytes", "model_box_to_decode_bytes", "model_cls_to_decode_bytes"),
        "decoded_prediction",
        False,
        False,
        False,
    ),
    StageDef(
        "postprocess",
        ("post_raw_decode_sec", "post_load_label_sec", "post_detection_metric_sec", "post_overlap_metric_sec"),
        ("post_raw_decode_bytes",),
        "detections",
        True,
        False,
        False,
    ),
)


def model_baseline_csv(model_name: str) -> Path:
    return LOCAL_BASELINE_DIR / f"baseline_FastSigmf_{model_name}_local_{DEVICE}.csv"


def stage_compute(row: dict[str, str], stage: StageDef) -> float:
    return sum(f(row.get(field)) for field in stage.compute_fields)


def stage_output_bytes(row: dict[str, str], stage: StageDef) -> float:
    return sum(f(row.get(field)) for field in stage.output_byte_fields)


def build_stage_stats() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_model: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        path = model_baseline_csv(model_name)
        rows = read_csv(path)
        model_entry = {
            "model_name": model_name,
            "source_csv": str(path),
            "sample_count": len(rows),
            "stages": {},
        }
        for index, stage in enumerate(STAGES):
            compute_values = [stage_compute(row, stage) for row in rows]
            byte_values = [stage_output_bytes(row, stage) for row in rows]
            compute_stats = stats(compute_values)
            byte_stats = stats(byte_values)
            stage_entry = {
                "stage_index": index,
                "stage": stage.name,
                "compute_sec": compute_stats,
                "output_bytes": byte_stats,
                "payload_type": stage.payload_type,
                "runtime_boundary_supported": stage.runtime_boundary_supported,
                "preprocess_internal_boundary": stage.preprocess_internal_boundary,
                "compression_allowed": stage.compression_allowed and not (
                    FORBID_PREPROCESS_INTERNAL_COMPRESSION and stage.preprocess_internal_boundary
                ),
                "compute_fields": stage.compute_fields,
                "output_byte_fields": stage.output_byte_fields,
            }
            model_entry["stages"][stage.name] = stage_entry
            for node in NODES:
                flat_rows.append(
                    {
                        "model_name": model_name,
                        "node": node,
                        "stage_index": index,
                        "stage": stage.name,
                        "payload_type": stage.payload_type,
                        "runtime_boundary_supported": stage.runtime_boundary_supported,
                        "preprocess_internal_boundary": stage.preprocess_internal_boundary,
                        "compression_allowed": stage_entry["compression_allowed"],
                        **{f"compute_sec_{k}": v for k, v in compute_stats.items()},
                        **{f"output_bytes_{k}": v for k, v in byte_stats.items()},
                    }
                )
        by_model[model_name] = model_entry
    return by_model, flat_rows


def detail_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(DISTRIBUTED_RESULTS_DIR.glob("*.csv")):
        if "summary" in path.name:
            continue
        for row in read_csv(path):
            if row.get("model_name") in MODEL_NAMES:
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def edge_payload_types(row: dict[str, str]) -> dict[str, str]:
    text = row.get("partition_json") or ""
    if not text:
        return {}
    try:
        partition = json.loads(text)
    except json.JSONDecodeError:
        return {}
    edges = partition.get("edges", {})
    return {
        "A_to_B": edges.get("A_to_B", {}).get("payload", ""),
        "B_to_C": edges.get("B_to_C", {}).get("payload", ""),
        "C_to_D": edges.get("C_to_D", {}).get("payload", ""),
        "D_to_A": edges.get("D_to_A", {}).get("payload", "detections"),
    }


def edge_policy(row: dict[str, str]) -> dict[str, str]:
    text = row.get("partition_json") or ""
    if not text:
        return {}
    try:
        partition = json.loads(text)
    except json.JSONDecodeError:
        return {}
    edges = partition.get("edges", {})
    return {
        "A_to_B": edges.get("A_to_B", {}).get("compression_policy", ""),
        "B_to_C": edges.get("B_to_C", {}).get("compression_policy", ""),
        "C_to_D": edges.get("C_to_D", {}).get("compression_policy", ""),
        "D_to_A": edges.get("D_to_A", {}).get("compression_policy", "forbidden"),
    }


def edge_values(row: dict[str, str], link: str) -> tuple[float, float]:
    prefix = link.lower()
    return f(row.get(f"{prefix}_payload_bytes")), f(row.get(f"{prefix}_send_sec"))


def build_link_bandwidth_stats(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[float, str], dict[str, Any]], dict[str, dict[str, float]]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    payload_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    send_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    edge_profile_rows: list[dict[str, Any]] = []

    for row in rows:
        model = row["model_name"]
        rate = f(row.get("rate_mbps"))
        comp = row.get("compression_label") or "nocompress"
        payload_types = edge_payload_types(row)
        policies = edge_policy(row)
        for link in LINKS:
            payload_bytes, send_sec = edge_values(row, link)
            if payload_bytes <= 0.0 or send_sec <= 0.0:
                continue
            effective_mbps = payload_bytes * 8.0 / send_sec / 1_000_000.0
            payload_type = payload_types.get(link, "")
            policy = policies.get(link, "")
            keys = [
                ("by_rate_link", rate, link),
                ("by_rate_compression_link", rate, comp, link),
                ("by_model_rate_compression_link", model, rate, comp, link),
                ("by_payload_compression", payload_type, comp),
            ]
            for key in keys:
                groups[key].append(effective_mbps)
                payload_groups[key].append(payload_bytes)
                send_groups[key].append(send_sec)
            edge_profile_rows.append(
                {
                    "model_name": model,
                    "rate_mbps": rate,
                    "compression_label": comp,
                    "link": link,
                    "payload_type": payload_type,
                    "compression_policy": policy,
                    "payload_bytes": payload_bytes,
                    "send_sec": send_sec,
                    "effective_mbps": effective_mbps,
                    "source_csv": row["_source_csv"],
                }
            )

    result: dict[str, Any] = defaultdict(dict)
    flat: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        group_type = str(key[0])
        group_values = key[1:]
        bw_stats = stats(values)
        payload_stats = stats(payload_groups[key])
        send_stats = stats(send_groups[key])
        record = {
            "group_type": group_type,
            "group_values": group_values,
            "effective_mbps": bw_stats,
            "payload_bytes": payload_stats,
            "send_sec": send_stats,
        }
        result[group_type][str(group_values)] = record
        flat.append(
            {
                "group_type": group_type,
                "group_values": "|".join(str(v) for v in group_values),
                **{f"effective_mbps_{k}": v for k, v in bw_stats.items()},
                **{f"payload_bytes_{k}": v for k, v in payload_stats.items()},
                **{f"send_sec_{k}": v for k, v in send_stats.items()},
            }
        )

    rate_link_stats: dict[tuple[float, str], dict[str, Any]] = {}
    for key, values in groups.items():
        if key[0] == "by_rate_link":
            _group_type, rate, link = key
            rate_link_stats[(float(rate), str(link))] = stats(values)

    compression_ratios = build_compression_ratios(edge_profile_rows)
    return dict(result), flat, rate_link_stats, compression_ratios


def build_compression_ratios(edge_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    nocompress: dict[tuple[str, float, str, str], float] = {}
    for row in edge_rows:
        if row["compression_label"] != "nocompress":
            continue
        nocompress[(row["model_name"], float(row["rate_mbps"]), row["link"], row["payload_type"])] = row["payload_bytes"]

    by_label_payload: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in edge_rows:
        label = row["compression_label"]
        if label == "nocompress":
            continue
        key = (row["model_name"], float(row["rate_mbps"]), row["link"], row["payload_type"])
        baseline = nocompress.get(key)
        if not baseline or baseline <= 0:
            continue
        ratio = row["payload_bytes"] / baseline
        # Keep only cases where compression actually affected the transmitted
        # payload. This avoids forbidden preprocess boundaries influencing the
        # estimated compression ratio.
        if ratio >= 0.98:
            continue
        by_label_payload[(label, row["payload_type"])].append(ratio)
        by_label[label].append(ratio)

    output: dict[str, dict[str, float]] = {}
    for (label, payload_type), values in by_label_payload.items():
        output.setdefault(label, {})[payload_type] = statistics.median(values)
    for label, values in by_label.items():
        output.setdefault(label, {})["__global__"] = statistics.median(values)
    return output


def stage_metric(stage_stats: dict[str, Any], model_name: str, stage_name: str, metric: str, stat_name: str) -> float:
    return stat_value(stage_stats[model_name]["stages"][stage_name][metric], stat_name)


def boundary_stage_for_cut(cut: int) -> StageDef:
    return STAGES[cut - 1]


def candidate_supported(cuts: tuple[int, int, int]) -> bool:
    if ALLOW_UNSUPPORTED_RUNTIME_BOUNDARIES:
        return True
    return all(boundary_stage_for_cut(cut).runtime_boundary_supported for cut in cuts)


def compression_policy_for_stage(stage: StageDef) -> str:
    allowed = stage.compression_allowed and not (
        FORBID_PREPROCESS_INTERNAL_COMPRESSION and stage.preprocess_internal_boundary
    )
    return "allowed" if allowed else "forbidden"


def compressed_bytes_for_boundary(
    raw_bytes: float,
    stage: StageDef,
    compression_label: str,
    compression_ratios: dict[str, dict[str, float]],
) -> tuple[float, str]:
    if compression_label == "nocompress":
        return raw_bytes, "nocompress"
    if compression_policy_for_stage(stage) != "allowed":
        return raw_bytes, "compression_forbidden_boundary"
    ratios = compression_ratios.get(compression_label, {})
    ratio = ratios.get(stage.payload_type, ratios.get("__global__"))
    if ratio is None:
        return raw_bytes, "missing_compression_ratio"
    return raw_bytes * ratio, f"estimated_ratio_{ratio:.6g}"


def link_bandwidth_mbps(rate_mbps: float, link: str, rate_link_stats: dict[tuple[float, str], dict[str, Any]]) -> float:
    stat = rate_link_stats.get((float(rate_mbps), link))
    if stat:
        value = stat_value(stat, BANDWIDTH_STAT)
        if value > 0:
            return value
    raise ValueError(f"Missing measured link rate: setting={rate_mbps}, link={link}")


def module_name_for(node: str, stages: list[str]) -> str:
    if len(stages) == 1:
        return stages[0]
    if stages[0].startswith("pre_") and stages[-1] == "pre_to_tensor":
        return "preprocess"
    return f"{stages[0]}__to__{stages[-1]}"


def build_partition_json_edges(
    cuts: tuple[int, int, int],
    edge_payload_bytes: dict[str, float],
    edge_comm_sec: dict[str, float],
) -> dict[str, Any]:
    edges: dict[str, Any] = {}
    for link, cut in zip(CHAIN_LINKS, cuts):
        src, dst = link.split("_to_")
        stage = boundary_stage_for_cut(cut)
        edges[link] = {
            "src": src,
            "dst": dst,
            "payload": stage.payload_type,
            "estimated_payload_bytes_per_sample": edge_payload_bytes[link],
            "estimated_comm_sec_per_sample": edge_comm_sec[link],
            "compression_policy": compression_policy_for_stage(stage),
        }
    edges["D_to_A"] = {
        "src": "D",
        "dst": "A",
        "payload": "detections",
        "estimated_payload_bytes_per_sample": edge_payload_bytes.get("D_to_A", 0.0),
        "estimated_comm_sec_per_sample": edge_comm_sec.get("D_to_A", 0.0),
        "compression_policy": "forbidden",
    }
    return edges


def make_candidate(
    model_name: str,
    rate_mbps: float,
    compression_label: str,
    cuts: tuple[int, int, int],
    stage_stats: dict[str, Any],
    rate_link_stats: dict[tuple[float, str], dict[str, Any]],
    compression_ratios: dict[str, dict[str, float]],
) -> dict[str, Any]:
    ranges = ((0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], len(STAGES)))
    node_compute: dict[str, float] = {}
    node_stages: dict[str, list[str]] = {}
    node_modules: dict[str, str] = {}
    modules: dict[str, Any] = {}
    for node, (start, end) in zip(NODES, ranges):
        stages = [stage.name for stage in STAGES[start:end]]
        compute = sum(stage_metric(stage_stats, model_name, stage.name, "compute_sec", STAGE_STAT) for stage in STAGES[start:end])
        module_name = module_name_for(node, stages)
        node_compute[node] = compute
        node_stages[node] = stages
        node_modules[node] = module_name
        modules[module_name] = {
            "node": node,
            "stages": stages,
            "estimated_compute_sec_per_sample": compute,
        }
        if "postprocess" in stages:
            modules[module_name]["postprocess_on_node"] = True

    edge_payload_bytes: dict[str, float] = {}
    edge_comm_sec: dict[str, float] = {}
    compression_notes: dict[str, str] = {}
    missing_compression_profile = False
    for link, cut in zip(CHAIN_LINKS, cuts):
        stage = boundary_stage_for_cut(cut)
        raw_bytes = stage_metric(stage_stats, model_name, stage.name, "output_bytes", BYTE_STAT)
        payload_bytes, note = compressed_bytes_for_boundary(raw_bytes, stage, compression_label, compression_ratios)
        if compression_label != "nocompress" and note == "missing_compression_ratio" and compression_policy_for_stage(stage) == "allowed":
            missing_compression_profile = True
        bw_mbps = link_bandwidth_mbps(rate_mbps, link, rate_link_stats)
        edge_payload_bytes[link] = payload_bytes
        edge_comm_sec[link] = payload_bytes * 8.0 / (bw_mbps * 1_000_000.0) if bw_mbps > 0 else math.inf
        compression_notes[link] = note

    d_stage = STAGES[-1]
    d_bytes = stage_metric(stage_stats, model_name, d_stage.name, "output_bytes", BYTE_STAT)
    d_bw_mbps = link_bandwidth_mbps(rate_mbps, "D_to_A", rate_link_stats)
    edge_payload_bytes["D_to_A"] = d_bytes
    edge_comm_sec["D_to_A"] = d_bytes * 8.0 / (d_bw_mbps * 1_000_000.0) if d_bw_mbps > 0 else math.inf
    compression_notes["D_to_A"] = "compression_forbidden_boundary"

    pipeline_components = {
        "A_compute": node_compute["A"],
        "A_to_B_comm": edge_comm_sec["A_to_B"],
        "B_compute": node_compute["B"],
        "B_to_C_comm": edge_comm_sec["B_to_C"],
        "C_compute": node_compute["C"],
        "C_to_D_comm": edge_comm_sec["C_to_D"],
        "D_compute": node_compute["D"],
        "D_to_A_comm": edge_comm_sec["D_to_A"],
    }
    primary = max(pipeline_components.values())
    secondary = sum(node_compute.values()) + sum(edge_comm_sec.values())
    return {
        "model_name": model_name,
        "rate_mbps": rate_mbps,
        "compression_label": compression_label,
        "cuts": list(cuts),
        "node_modules": node_modules,
        "modules": modules,
        "edges": build_partition_json_edges(cuts, edge_payload_bytes, edge_comm_sec),
        "node_compute_sec": node_compute,
        "edge_payload_bytes": edge_payload_bytes,
        "edge_comm_sec": edge_comm_sec,
        "compression_notes": compression_notes,
        "missing_compression_profile": missing_compression_profile,
        "pipeline_components_sec": pipeline_components,
        "primary_pipeline_time_per_sample_sec": primary,
        "secondary_e2e_latency_sec": secondary,
        "total_compute_sec": sum(node_compute.values()),
        "total_comm_sec": sum(edge_comm_sec.values()),
        "total_payload_bytes": sum(edge_payload_bytes.values()),
        "max_node_compute_sec": max(node_compute.values()),
        "compute_imbalance_sec": max(node_compute.values()) - min(node_compute.values()),
    }


def all_supported_cuts() -> list[tuple[int, int, int]]:
    cuts = list(itertools.combinations(range(1, len(STAGES)), 3))
    return [cut for cut in cuts if candidate_supported(cut)]


def rank_candidates(
    candidates: list[dict[str, Any]],
    allow_missing_compression_profile: bool,
) -> list[dict[str, Any]]:
    filtered = [
        candidate
        for candidate in candidates
        if allow_missing_compression_profile or not candidate.get("missing_compression_profile")
    ]
    filtered.sort(
        key=lambda item: (
            item["primary_pipeline_time_per_sample_sec"],
            item["secondary_e2e_latency_sec"],
            item["total_comm_sec"],
            item["max_node_compute_sec"],
            item["compute_imbalance_sec"],
            item["total_payload_bytes"],
        )
    )
    for index, candidate in enumerate(filtered[:TOP_K], start=1):
        candidate["rank"] = index
    return filtered[:TOP_K]


def flatten_candidate(candidate: dict[str, Any], mode: str) -> dict[str, Any]:
    row = {
        "mode": mode,
        "rank": candidate.get("rank"),
        "model_name": candidate["model_name"],
        "rate_mbps": candidate["rate_mbps"],
        "compression_label": candidate["compression_label"],
        "cuts": json.dumps(candidate["cuts"], separators=(",", ":")),
        "primary_pipeline_time_per_sample_sec": candidate["primary_pipeline_time_per_sample_sec"],
        "secondary_e2e_latency_sec": candidate["secondary_e2e_latency_sec"],
        "total_compute_sec": candidate["total_compute_sec"],
        "total_comm_sec": candidate["total_comm_sec"],
        "total_payload_bytes": candidate["total_payload_bytes"],
        "max_node_compute_sec": candidate["max_node_compute_sec"],
        "compute_imbalance_sec": candidate["compute_imbalance_sec"],
        "missing_compression_profile": candidate["missing_compression_profile"],
    }
    for node in NODES:
        row[f"{node}_module"] = candidate["node_modules"][node]
        row[f"{node}_compute_sec"] = candidate["node_compute_sec"][node]
    for link in LINKS:
        row[f"{link}_payload_bytes"] = candidate["edge_payload_bytes"].get(link, 0.0)
        row[f"{link}_comm_sec"] = candidate["edge_comm_sec"].get(link, 0.0)
        row[f"{link}_compression_note"] = candidate["compression_notes"].get(link, "")
    return row


def build_rankings(
    stage_stats: dict[str, Any],
    rate_link_stats: dict[tuple[float, str], dict[str, Any]],
    compression_ratios: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    supported_cuts = all_supported_cuts()
    nocompress: dict[str, Any] = {}
    compressed: dict[str, Any] = {}
    flat_nocompress: list[dict[str, Any]] = []
    flat_compressed: list[dict[str, Any]] = []

    compression_labels = sorted(label for label in compression_ratios if label != "nocompress")
    for model_name in MODEL_NAMES:
        for rate_mbps in RATE_GROUPS_MBPS:
            base_candidates = [
                make_candidate(model_name, rate_mbps, "nocompress", cuts, stage_stats, rate_link_stats, compression_ratios)
                for cuts in supported_cuts
            ]
            top = rank_candidates(base_candidates, allow_missing_compression_profile=True)
            nocompress.setdefault(model_name, {})[str(rate_mbps)] = {
                "rate_mbps": rate_mbps,
                "compression_label": "nocompress",
                "candidate_count": len(base_candidates),
                "top_partitions": top,
            }
            flat_nocompress.extend(flatten_candidate(candidate, "nocompress") for candidate in top)

            for label in compression_labels:
                candidates = [
                    make_candidate(model_name, rate_mbps, label, cuts, stage_stats, rate_link_stats, compression_ratios)
                    for cuts in supported_cuts
                ]
                top_compressed = rank_candidates(candidates, allow_missing_compression_profile=False)
                compressed.setdefault(model_name, {}).setdefault(str(rate_mbps), {})[label] = {
                    "rate_mbps": rate_mbps,
                    "compression_label": label,
                    "skipped": not bool(top_compressed),
                    "reason": "" if top_compressed else "no_candidate_with_available_compression_profile",
                    "candidate_count": len(candidates),
                    "top_partitions": top_compressed,
                }
                flat_compressed.extend(flatten_candidate(candidate, "compressed") for candidate in top_compressed)

    return nocompress, compressed, flat_nocompress, flat_compressed


def selected_partition_for_model(rankings: dict[str, Any], model_name: str, compressed: bool) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if not compressed:
        for rate_block in rankings.get(model_name, {}).values():
            candidates.extend(rate_block.get("top_partitions", []))
    else:
        for rate_block in rankings.get(model_name, {}).values():
            for comp_block in rate_block.values():
                candidates.extend(comp_block.get("top_partitions", []))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["primary_pipeline_time_per_sample_sec"],
            item["secondary_e2e_latency_sec"],
            item["total_comm_sec"],
        ),
    )


def make_partition_file(rankings: dict[str, Any], compressed: bool) -> dict[str, Any]:
    family_name = "asym_yolo_v8_f20t20_c100_jetson_comm_aware"
    output = {
        "schema_version": 2,
        "plan_type": "jetson_comm_aware_asym_yolo_partition",
        "platform": PLATFORM,
        "device": DEVICE,
        "topology": "A->B->C->D->A",
        "objective": {
            "primary": "minimize max(node_compute_sec, edge_comm_sec)",
            "secondary": "minimize sum(node_compute_sec) + sum(edge_comm_sec)",
            "stage_stat": STAGE_STAT,
            "byte_stat": BYTE_STAT,
            "bandwidth_stat": BANDWIDTH_STAT,
        },
        "compression_mode": "compressed_unfiltered" if compressed else "nocompress",
        "families": {
            family_name: {
                "graph": "ordered asym yolo stages with communication-aware contiguous chain partition",
                "models": {},
            }
        },
    }
    models = output["families"][family_name]["models"]
    for model_name in MODEL_NAMES:
        selected = selected_partition_for_model(rankings, model_name, compressed)
        if selected is None:
            models[model_name] = {"selected_partition": None, "candidate_partitions": []}
            continue
        candidate_partitions: list[dict[str, Any]] = []
        if not compressed:
            for rate_block in rankings.get(model_name, {}).values():
                candidate_partitions.extend(rate_block.get("top_partitions", []))
        else:
            for rate_block in rankings.get(model_name, {}).values():
                for comp_block in rate_block.values():
                    candidate_partitions.extend(comp_block.get("top_partitions", []))
        candidate_partitions = sorted(
            candidate_partitions,
            key=lambda item: (
                item["primary_pipeline_time_per_sample_sec"],
                item["secondary_e2e_latency_sec"],
                item["total_comm_sec"],
            ),
        )[:TOP_K]
        models[model_name] = {
            "selected_partition": selected,
            "candidate_partitions": candidate_partitions,
            "node_modules": selected["node_modules"],
            "modules": selected["modules"],
            "edges": selected["edges"],
            "total_estimated_compute_sec_per_sample": selected["total_compute_sec"],
            "primary_pipeline_time_per_sample_sec": selected["primary_pipeline_time_per_sample_sec"],
            "secondary_e2e_latency_sec": selected["secondary_e2e_latency_sec"],
        }
    return output


def main() -> None:
    global OUTPUT_DIR, LOCAL_BASELINE_DIR, DISTRIBUTED_RESULTS_DIR
    parser = argparse.ArgumentParser(description='Rebuild offline partition rankings from supplied profiles or new measurements.')
    parser.add_argument('--from-profiles', type=Path, default=PROJECT_ROOT / 'configs')
    parser.add_argument('--measurements', action='store_true', help='Build profiles from measured stage and transfer CSVs instead')
    parser.add_argument('--local-baseline-dir', type=Path, default=LOCAL_BASELINE_DIR)
    parser.add_argument('--distributed-results-dir', type=Path, default=DISTRIBUTED_RESULTS_DIR)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output.resolve()
    LOCAL_BASELINE_DIR = args.local_baseline_dir.resolve()
    DISTRIBUTED_RESULTS_DIR = args.distributed_results_dir.resolve()
    if OUTPUT_DIR == args.from_profiles.resolve():
        raise ValueError('Use a separate output directory to preserve the reference profiles.')
    analysis_start = time.perf_counter()
    READ_AUDIT.clear()
    if args.measurements:
        print_input_folder_snapshots()
        stage_stats, stage_flat = build_stage_stats()
        details = detail_rows()
        if not details:
            raise ValueError('No distributed transfer measurements found; no output written.')
        link_stats, link_flat, rate_link_stats, compression_ratios = build_link_bandwidth_stats(details)
    else:
        profile_dir = args.from_profiles
        stage_stats = json.loads((profile_dir / 'asym_jetson_stage_compute_payload_stats.json').read_text())['models']
        links = json.loads((profile_dir / 'asym_jetson_link_bandwidth_stats.json').read_text())
        link_stats, compression_ratios = links['link_bandwidth'], links['compression_ratios']
        rate_link_stats = {ast.literal_eval(k): v['effective_mbps'] for k,v in link_stats['by_rate_link'].items()}
        stage_flat = read_csv(profile_dir / 'asym_jetson_stage_compute_payload_stats.csv')
        link_flat = read_csv(profile_dir / 'asym_jetson_link_bandwidth_stats.csv')
    missing_models = set(MODEL_NAMES) - set(stage_stats)
    if missing_models:
        raise ValueError(f'Missing stage measurements for {sorted(missing_models)}')
    for model in MODEL_NAMES:
        for stage in STAGES:
            measured = stage_stats[model]['stages'].get(stage.name)
            if not measured or measured['compute_sec']['count'] <= 0 or measured['output_bytes']['count'] <= 0:
                raise ValueError(f'Missing stage measurements: {model}/{stage.name}')
    for rate in RATE_GROUPS_MBPS:
        for link in LINKS:
            link_bandwidth_mbps(rate, link, rate_link_stats)
    required_compressors = {'quant_int8', 'quant_int4', 'topk_0p25', 'topk_0p5', 'topk_0p75', 'llmint8_o0p01_fp16_int8'}
    missing = required_compressors - set(compression_ratios)
    if missing:
        raise ValueError(f'Missing measured compression ratios: {sorted(missing)}')
    nocompress_rankings, compressed_rankings, flat_nocompress, flat_compressed = build_rankings(
        stage_stats,
        rate_link_stats,
        compression_ratios,
    )

    metadata = {
        "platform": PLATFORM,
        "device": DEVICE,
        "models": MODEL_NAMES,
        "rates_mbps": RATE_GROUPS_MBPS,
        "stage_stat": STAGE_STAT,
        "byte_stat": BYTE_STAT,
        "bandwidth_stat": BANDWIDTH_STAT,
        "top_k": TOP_K,
        "notes": [
            "Primary ranking metric is pipeline throughput bottleneck: max(node_compute_sec, edge_comm_sec).",
            "Secondary ranking metric is single-sample E2E estimate: sum(node_compute_sec)+sum(edge_comm_sec).",
            "Compression byte ratios are inferred only from distributed cases where payload bytes actually decreased.",
            "Unsupported runtime boundaries are excluded unless ALLOW_UNSUPPORTED_RUNTIME_BOUNDARIES is set True.",
        ],
        "input_folders": {
            "local_baseline": csv_directory_snapshot(LOCAL_BASELINE_DIR),
            "distributed_results": csv_directory_snapshot(DISTRIBUTED_RESULTS_DIR),
            "output": csv_directory_snapshot(OUTPUT_DIR),
        },
        "read_audit": list(READ_AUDIT),
    }

    write_json(OUTPUT_DIR / "asym_jetson_stage_compute_payload_stats.json", {"metadata": metadata, "models": stage_stats})
    write_csv(
        OUTPUT_DIR / "asym_jetson_stage_compute_payload_stats.csv",
        stage_flat,
        [
            "model_name",
            "node",
            "stage_index",
            "stage",
            "payload_type",
            "runtime_boundary_supported",
            "preprocess_internal_boundary",
            "compression_allowed",
            "compute_sec_count",
            "compute_sec_min",
            "compute_sec_max",
            "compute_sec_mean",
            "compute_sec_median",
            "compute_sec_variance",
            "compute_sec_std",
            "compute_sec_p90",
            "compute_sec_p95",
            "output_bytes_count",
            "output_bytes_min",
            "output_bytes_max",
            "output_bytes_mean",
            "output_bytes_median",
            "output_bytes_variance",
            "output_bytes_std",
            "output_bytes_p90",
            "output_bytes_p95",
        ],
    )
    write_json(
        OUTPUT_DIR / "asym_jetson_link_bandwidth_stats.json",
        {"metadata": metadata, "link_bandwidth": link_stats, "compression_ratios": compression_ratios},
    )
    write_csv(
        OUTPUT_DIR / "asym_jetson_link_bandwidth_stats.csv",
        link_flat,
        [
            "group_type",
            "group_values",
            "effective_mbps_count",
            "effective_mbps_min",
            "effective_mbps_max",
            "effective_mbps_mean",
            "effective_mbps_median",
            "effective_mbps_variance",
            "effective_mbps_std",
            "effective_mbps_p90",
            "effective_mbps_p95",
            "payload_bytes_count",
            "payload_bytes_min",
            "payload_bytes_max",
            "payload_bytes_mean",
            "payload_bytes_median",
            "payload_bytes_variance",
            "payload_bytes_std",
            "payload_bytes_p90",
            "payload_bytes_p95",
            "send_sec_count",
            "send_sec_min",
            "send_sec_max",
            "send_sec_mean",
            "send_sec_median",
            "send_sec_variance",
            "send_sec_std",
            "send_sec_p90",
            "send_sec_p95",
        ],
    )
    ranking_fields = [
        "mode",
        "rank",
        "model_name",
        "rate_mbps",
        "compression_label",
        "cuts",
        "primary_pipeline_time_per_sample_sec",
        "secondary_e2e_latency_sec",
        "total_compute_sec",
        "total_comm_sec",
        "total_payload_bytes",
        "max_node_compute_sec",
        "compute_imbalance_sec",
        "missing_compression_profile",
        "A_module",
        "A_compute_sec",
        "B_module",
        "B_compute_sec",
        "C_module",
        "C_compute_sec",
        "D_module",
        "D_compute_sec",
        "A_to_B_payload_bytes",
        "A_to_B_comm_sec",
        "A_to_B_compression_note",
        "B_to_C_payload_bytes",
        "B_to_C_comm_sec",
        "B_to_C_compression_note",
        "C_to_D_payload_bytes",
        "C_to_D_comm_sec",
        "C_to_D_compression_note",
        "D_to_A_payload_bytes",
        "D_to_A_comm_sec",
        "D_to_A_compression_note",
    ]
    write_json(
        OUTPUT_DIR / "asym_jetson_partition_rankings_nocompress.json",
        {"metadata": metadata, "rankings": nocompress_rankings},
    )
    write_csv(OUTPUT_DIR / "asym_jetson_partition_rankings_nocompress.csv", flat_nocompress, ranking_fields)
    write_json(
        OUTPUT_DIR / "asym_jetson_partition_rankings_compressed.json",
        {"metadata": metadata, "rankings": compressed_rankings},
    )
    write_csv(OUTPUT_DIR / "asym_jetson_partition_rankings_compressed.csv", flat_compressed, ranking_fields)
    write_json(
        OUTPUT_DIR / "asym_module_boundary_partitions_jetson_nocompress_best.json",
        make_partition_file(nocompress_rankings, compressed=False),
    )
    write_json(
        OUTPUT_DIR / "asym_module_boundary_partitions_jetson_compressed_best.json",
        make_partition_file(compressed_rankings, compressed=True),
    )

    print(f"Wrote asym Jetson partition profiles to {OUTPUT_DIR}")
    print(f"nocompress top rows: {len(flat_nocompress)}")
    print(f"compressed top rows: {len(flat_compressed)}")
    print_read_audit()
    if PRINT_INPUT_AUDIT:
        output_snapshot = csv_directory_snapshot(OUTPUT_DIR)
        print(
            f"[output-folder] path={output_snapshot['path']} csv_count={output_snapshot['csv_count']} "
            f"total_csv_bytes={output_snapshot['total_csv_bytes']} latest_csv_mtime={output_snapshot['latest_csv_mtime']}"
        )
    print(f"analysis_total_sec: {time.perf_counter() - analysis_start:.6f}")


if __name__ == "__main__":
    main()
