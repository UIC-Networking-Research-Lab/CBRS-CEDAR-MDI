# Reference results

**[Open the experiment report with all four experiment sections and figures](REPORT.md).**
This page is the source-data index; the report explains the comparisons and key findings.

## `paper/` – data behind the paper's tables and figures

| File | Paper item | Content |
|---|---|---|
| `yolo_cpu_detector_results.csv` | Table I | 9 YOLO models, local Jetson CPU: delay, throughput, overlap precision/recall/F1, mAP@0.5 |
| `yolo_cpu_stage_cost_data.csv` | CPU stage-cost figure | mean CPU time per pipeline stage for the 9 YOLO models |
| `yolo_module_boundary_summary.csv` | MDI exploration | mean payload per frame and MDI / local delay ratios at high, medium and low bandwidth |
| `yolo_mdi_vs_local_results.csv` | MDI exploration | uncompressed MDI vs. local delay, throughput and quality (full test split) |
| `yolo_mdi_bandwidth_comparison_data.csv` | YOLO MDI figure | 9 models × 7 compression settings × 3 bandwidths, first 100 test frames |
| `cedar_yolo_sweep_data.csv` | CEDAR-YOLO sweep figure | 108 configurations: scale, grid, coverage, mAP@0.5, overlap F1, parameters |
| `cedar_yolo_local_comparison_data.csv` | CEDAR-YOLO vs. YOLOv8 figure | parameters, model time and quality vs. YOLOv8 at the same scale |
| `cedar_yolo_cpu_results.csv` | – | raw-I/Q local CPU results of the MDI checkpoints (`trained_models_mdi/`) next to YOLOv8 |
| `cedar_yolo_mdi_comparison_data.csv` | CEDAR-MDI figure | 3 models × 7 compression settings × 3 bandwidths × top-3 partitions, plus the local baseline |
| `cedar_mdi_timing_summary.csv` | CEDAR-MDI text | throughput and task-delay ratios of the runs within 0.01 F1 of local execution |

The `*_source*` / `source_csv` columns record which experiment output each row was
derived from. These paths refer to the authors' result folders, not to this repository.

## `training_logs/` – YOLO / RT-DETR training records

The Ultralytics `args.yaml` (full training configuration) and `results.csv` (per-epoch
losses and validation metrics) of each released checkpoint in `trained_models/`.

## Metric definitions

| Metric | Definition |
|---|---|
| mAP@0.5 | box detection mean average precision at IoU 0.5 over LTE and radar |
| Overlap precision / recall / F1 | frame-level decision "radar overlaps LTE in frequency", with overlap as the positive class; computed from confusion counts pooled over the frames |
| Delay / task delay | per-frame time from raw SigMF input to detections (for MDI: from admission at node A until the result returns) |
| Throughput / amortized time | frames divided by total run time / its inverse, including pipeline fill and drain |

The test split has 848 frames (558 overlapping, 290 non-overlapping). The 100-frame
compression runs use the first 100 test frames, all of which overlap.
