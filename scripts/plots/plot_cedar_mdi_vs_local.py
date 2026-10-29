"""
Plot AsymYOLO distributed compressor sweep against local AsymYOLO baseline.

Defaults:
    local asym baseline:        ./local_baseline_results_by_stage_cost_asym_yolo
    distributed asym compressor ./distributed_stage_sweep_results_asym

The script writes:
    - one combined comparison CSV
    - one bandwidth-filtered CSV per network rate
    - one PNG per comparison metric, grouped by bandwidth
    - stage/communication breakdown PNGs with local-baseline delay overlays
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.plots.plot_yolo_mdi_vs_local import (
    METRICS,
    ResultRow,
    build_stage_comm_plots,
    flatten_rows,
    latest_summary_csv,
    load_baseline_quality,
    load_distributed,
    model_family,
    read_csv_rows,
    to_float,
    write_bandwidth_subset_csvs,
    write_csv_rows,
    plot_metric_by_bandwidth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "local_baseline_results_by_stage_cost_asym_yolo"
DISTRIBUTED_ASYM_DIR = PROJECT_ROOT / "distributed_stage_sweep_results_asym"
OUTPUT_DIR = PROJECT_ROOT / "asym_yolo_distributed_vs_baseline_plots"
BASELINE_DEVICE = "cpu"


def baseline_model_from_csv(path: Path) -> str:
    prefix = "baseline_FastSigmf_"
    suffix = "_local_cpu.csv"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected CPU baseline csv name: {path.name}")
    return name[len(prefix) : -len(suffix)]


def load_asym_cpu_baseline(baseline_dir: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    pattern = "baseline_FastSigmf_*_local_cpu.csv"
    for csv_path in sorted(baseline_dir.glob(pattern)):
        model_name = baseline_model_from_csv(csv_path)
        csv_rows = read_csv_rows(csv_path)
        sample_count = len(csv_rows)
        if sample_count == 0:
            continue

        completed_at = [to_float(row.get("completed_at_sec"), default=np.nan) for row in csv_rows]
        valid_completed = [v for v in completed_at if np.isfinite(v)]
        if valid_completed:
            total_sec = max(valid_completed)
        else:
            total_sec = sum(to_float(row.get("end_to_end_sec")) for row in csv_rows)

        metrics: dict[str, float] = {
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
                method="local cpu baseline",
                source="local baseline",
                compression_label="",
                rate_mbps=None,
                sample_count=sample_count,
                metrics=metrics,
            )
        )

    if not rows:
        raise FileNotFoundError(f"No {pattern} found under {baseline_dir}")
    return rows


def build_plots(
    baseline_dir: Path,
    distributed_summary: Path,
    output_dir: Path,
) -> None:
    baseline_rows = load_asym_cpu_baseline(baseline_dir)
    distributed_rows = load_distributed(
        distributed_summary,
        source="asym distributed compressor",
        method_prefix="asym distributed",
        include_compression=True,
        include_partitions=True,
    )
    all_rows = baseline_rows + distributed_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = output_dir / "asym_yolo_distributed_vs_baseline_comparison.csv"
    write_csv_rows(comparison_csv, flatten_rows(all_rows))

    by_bandwidth_dir = output_dir / "by_bandwidth"
    by_bandwidth_dir.mkdir(parents=True, exist_ok=True)
    write_bandwidth_subset_csvs(all_rows, output_dir)

    for metric, title, ylabel in METRICS:
        plot_metric_by_bandwidth(
            all_rows,
            metric=metric,
            title=f"AsymYOLO Distributed Compressor vs Local Baseline: {title}",
            ylabel=ylabel,
            output_path=by_bandwidth_dir / f"{metric}.png",
        )
    build_stage_comm_plots(all_rows, output_dir / "stage_comm")

    print(f"[plot] baseline dir: {baseline_dir}")
    print(f"[plot] baseline device: {BASELINE_DEVICE}")
    print(f"[plot] distributed compressor summary: {distributed_summary}")
    print(f"[plot] comparison csv: {comparison_csv}")
    print(f"[plot] bandwidth-grouped figures written to: {by_bandwidth_dir}")
    print(f"[plot] stage/communication figures written to: {output_dir / 'stage_comm'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AsymYOLO distributed compressor vs local CPU AsymYOLO baseline metrics."
    )
    parser.add_argument("--baseline_dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--distributed_summary", type=Path, default=None)
    parser.add_argument("--distributed_dir", type=Path, default=DISTRIBUTED_ASYM_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed_summary = args.distributed_summary or latest_summary_csv(args.distributed_dir)
    build_plots(
        baseline_dir=args.baseline_dir,
        distributed_summary=distributed_summary,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
