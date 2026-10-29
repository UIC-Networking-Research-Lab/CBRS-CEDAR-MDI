# Paper experiment report

Results accompanying **CEDAR-MDI: CBRS-Oriented Edge-Distributed Architecture for
Radar Detection with Model-Distributed Inference** (manuscript under submission).
All figures below are included in the repository and display without downloading
datasets or weights. Numerical records are in [`paper/`](paper/).

| Experiment | Finding | Details |
|---|---|---|
| Original YOLO | High radar/LTE detection quality on the Jetson platform | [1. Local detection](#1-original-yolo-local-detection) |
| CEDAR-YOLO | Fewer parameters and less model computation with a single-scale architecture | [2. Architecture evaluation](#2-cedar-yolo-architecture-evaluation) |
| Original-YOLO MDI | Activation transfers make per-frame latency higher than local execution | [3. Original-YOLO MDI](#3-original-yolo-mdi) |
| CEDAR-MDI | Up to **3.27× local throughput** with unchanged overlap F1 at the best setting | [4. CEDAR-MDI](#4-cedar-mdi) |

## Setup and metrics

| Experiment | Environment |
|---|---|
| Original-YOLO and CEDAR-YOLO training; CEDAR architecture/quality sweep | PC GPU, Conda `torch270_py312_cuda`, Python 3.12, PyTorch 2.7.1+cu128 |
| Local speed baselines and stage timings | One Jetson Orin Nano CPU, FP32, L4T r36.4.0, `/opt/venv`, Python 3.12, PyTorch 2.7.0 |
| Original-YOLO MDI and CEDAR-MDI | Multiple Jetsons using the same Jetson environment, CPU FP32 |

Full [recommended/tested library versions](../README.md#experiment-environments)
are documented separately for PC and Jetson. The PC CEDAR quality sweep uses
the later checkpoints; Jetson timings and MDI use the earlier Jetson edition.

The edge testbed uses four NVIDIA Jetson Orin Nano 8 GB devices with CPU FP32
execution over Wi-Fi. Token-bucket parameters create high, medium and low bandwidth
environments. Local baselines run on one device. The fixed test split contains
848 frames; the original-YOLO compression sweep uses its first 100 frames, all
overlap-positive. Other evaluations use the full test split.

| Metric | Meaning | Better |
|---|---|---|
| Parameters | Number of model parameters | Lower |
| Model execution time | Detector computation, excluding I/Q preprocessing and postprocessing | Lower |
| Task delay | Time from one frame entering the raw-I/Q pipeline until its detections return | Lower |
| Inference time per frame | Total run time / completed frames = 1 / throughput; called amortized completion time in the figures | Lower |
| Overlap F1 | Frame-level decision that radar and LTE overlap in frequency | Higher |
| mAP@0.5 | Radar/LTE bounding-box detection quality at IoU 0.5 | Higher |

The later PC architecture/quality sweep and the earlier Jetson/MDI experiments use
different CEDAR checkpoint editions. Each CEDAR-MDI comparison uses the **same
checkpoint for local and distributed execution**. See the [weight guide](../docs/weights.md).

## 1. Original YOLO: local detection

YOLOv8, YOLO11 and YOLO26 detect radar/LTE signals with high quality. Larger models
increase computation without a corresponding improvement in overlap F1 on this dataset.

| Model | Task delay (s) ↓ | Throughput (frames/s) ↑ | Overlap F1 ↑ | mAP@0.5 ↑ |
|---|---:|---:|---:|---:|
| YOLOv8n | 1.181 | 0.846 | 0.997 | 0.995 |
| YOLOv8s | 1.489 | 0.671 | 0.999 | 0.995 |
| YOLOv8x | 4.123 | 0.243 | 0.995 | 0.995 |
| YOLO11n | 1.142 | 0.875 | 0.997 | 0.995 |
| YOLO11s | 1.452 | 0.688 | 1.000 | 0.995 |
| YOLO11x | 3.839 | 0.260 | 0.995 | 0.995 |
| YOLO26n | 1.226 | 0.815 | 0.998 | 0.994 |
| YOLO26s | 1.593 | 0.628 | 0.995 | 0.995 |
| YOLO26x | 4.168 | 0.240 | 0.991 | 0.991 |

![Original YOLO CPU time by pipeline stage](../docs/figures/yolo_cpu_stage_cost_breakdown.png)

**Reading the figure:** left, I/Q preprocessing and postprocessing; right, detector
computation broken down by stage. Preprocessing accounts for much of the small-model
runtime, while model computation dominates the extra-large variants. Stage profiling
and the end-to-end table are separate timing runs.

Sources: [detector results](paper/yolo_cpu_detector_results.csv),
[stage costs](paper/yolo_cpu_stage_cost_data.csv).
Reproduction: [original-YOLO training and inference](../docs/1_yolo.md).

## 2. CEDAR-YOLO: architecture evaluation

CEDAR-YOLO removes the multi-scale fusion neck and uses a single-scale detection
path. The comparison below uses a 20 × 20 frequency/time grid with 100% regression
coverage. Parameter counts fall by 35–57%, and Jetson model execution time falls
by 16–33% across the three model sizes.

![CEDAR-YOLO versus original YOLOv8: size, model time and quality](../docs/figures/cedar_yolo_local_comparison.png)

**Reading the figure:** left, parameter count and model execution time normalized
to YOLOv8 at the same model size; right, mAP@0.5 and overlap F1. The timing bars use
the earlier checkpoints measured on Jetson; the quality bars use the later sweep checkpoints evaluated on the PC GPU.
The tables below keep these two evaluations separate.

**Model size and Jetson timing** (original YOLOv8 → CEDAR-YOLO):

| Model size | Parameters (M) ↓ | Model execution time (s) ↓ |
|---|---:|---:|
| Nano | 3.01 → 1.85 | 0.264 → 0.178 |
| Small | 11.14 → 7.27 | 0.582 → 0.431 |
| Extra-large | 68.15 → 29.17 | 3.076 → 2.593 |

**Detection quality** (the two CEDAR checkpoint editions):

| Model size | YOLOv8 mAP@0.5 / overlap F1 | Later CEDAR sweep mAP@0.5 / overlap F1 | Earlier CEDAR Jetson mAP@0.5 / overlap F1 |
|---|---:|---:|---:|
| Nano | 0.995 / 0.997 | 0.995 / 0.999 | 0.895 / 0.995 |
| Small | 0.995 / 0.999 | 0.995 / 0.996 | 0.995 / 0.999 |
| Extra-large | 0.995 / 0.995 | 0.995 / 1.000 | 0.995 / 1.000 |

### Feature-grid and regression-coverage sweep

![CEDAR-YOLO quality across frequency/time grids and coverage levels](../docs/figures/cedar_yolo_grid_sweep.png)

Each cell is one trained configuration. Rows group nano, small and extra-large
models; within each group, frequency cells vary vertically and time cells vary
horizontally. Coverage controls the box-regression distribution range, not dataset
coverage. The left panel reports box mAP@0.5 and the right panel reports overlap F1.
The 20 × 20 grid at 100% coverage gives high scores for all three model sizes.

Sources: [local comparison](paper/cedar_yolo_local_comparison_data.csv),
[full architecture sweep](paper/cedar_yolo_sweep_data.csv),
[earlier Jetson checkpoint results](paper/cedar_yolo_cpu_results.csv).
Reproduction: [CEDAR-YOLO training and inference](../docs/2_cedar_yolo.md).

## 3. Original-YOLO MDI

The module-boundary baseline places preprocessing on node A and the backbone,
neck and detection head on nodes B–D. Multiple feature maps cross the later node
boundaries. Compression reduces transfer cost, but every tested setting in this
compression sweep has higher task delay than its local baseline.

![Original-YOLO partition, compression quality and task delay](../docs/figures/yolo_mdi_bandwidth_comparison.png)

**Reading the figure:** top left, the node assignments and transmitted tensors;
top right, overlap F1 at high bandwidth; bottom, task delay for high, medium and
low bandwidth. Dashed lines mark the local baselines. INT8 largely retains
overlap quality; aggressive Top-k compression can substantially reduce it.

The separate full-test, uncompressed comparison quantifies the payload and delay:

| Model | Local task delay (s) | MDI task delay, high bandwidth (s) | Transmitted payload (MB/frame) |
|---|---:|---:|---:|
| YOLOv8n | 1.181 | 2.573 | 10.66 |
| YOLOv8s | 1.489 | 3.846 | 16.39 |
| YOLOv8x | 4.123 | 11.917 | 31.55 |

Sources: [compression sweep](paper/yolo_mdi_bandwidth_comparison_data.csv),
[full-test local/MDI comparison for all nine models](paper/yolo_mdi_vs_local_results.csv),
[payload and delay ratios](paper/yolo_module_boundary_summary.csv).
Reproduction: [original-YOLO MDI](../docs/3_yolo_mdi.md).

## 4. CEDAR-MDI

CEDAR-MDI evaluates partition and node-assignment policies offline using measured
compute and communication costs, then deploys the top three policies per model,
bandwidth environment and compressor. A single-scale stage chain keeps boundary
transfers small and supports pipelined execution.

![CEDAR-MDI completed-frame time and overlap F1 across bandwidth environments](../docs/figures/cedar_yolo_mdi_comparison.png)

**Reading the figure:** left, inference time per completed frame, where lower is
better; right, overlap F1 for the same configurations. Rows correspond to high,
medium and low bandwidth. `n`, `s` and `x` denote model size. `R1–R3` are candidate
ranks within each setting; the actual partitions can change with the model,
bandwidth and compressor. Dashed lines show local completion time.

The summary retains policies with overlap F1 within 0.01 of local execution,
excluding configurations with poor detection quality. The figure and source CSV
include all policies, including those excluded from this summary.

| Model | Local time per frame (s/frame) | CEDAR-MDI time per frame (s/frame) | Throughput / local | Policies retained |
|---|---:|---:|---:|---:|
| CEDAR-YOLO-n | 1.054 | 0.884–0.902 | 1.17–1.19× | 57 / 63 |
| CEDAR-YOLO-s | 1.310 | 0.888–0.903 | 1.45–1.47× | 57 / 63 |
| CEDAR-YOLO-x | 3.455 | 1.055–2.495 | **1.38–3.27×** | 45 / 63 |

**Best measured setting:** CEDAR-YOLO-x, high bandwidth, INT8, candidate R3.

| Metric | Local | CEDAR-MDI |
|---|---:|---:|
| Inference time per frame (s/frame) ↓ | 3.455 | **1.055** |
| Throughput (frames/s) ↑ | 0.289 | **0.948** |
| Overlap F1 ↑ | 1.000 | **1.000** |
| Mean end-to-end task delay (s) ↓ | 3.454 | 6.292 |

**CEDAR-MDI achieves 3.27× local throughput while preserving overlap F1 in this
setting.** Pipeline parallelism improves the rate of completed frames. Each
individual frame still traverses several nodes, so its end-to-end task delay is
higher than local execution.

Sources: [all measured local/MDI configurations](paper/cedar_yolo_mdi_comparison_data.csv),
[quality-filtered timing summary](paper/cedar_mdi_timing_summary.csv).
Reproduction: [CEDAR-MDI profiling, optimization and deployment](../docs/4_cedar_mdi.md).

## Data and reproduction links

- [Complete source-data index and metric definitions](README.md)
- [Quick start and experiment commands](../README.md#quick-start-use-our-pngs-and-pretrained-models)
- [Dataset, supplied PNGs and fixed splits](../docs/data.md)
- [Preprocessing parameters and library-version compatibility](../docs/preprocessing.md)
- [Checkpoint editions and downloads](../docs/weights.md)

These are the recorded paper results. Reproduction commands write new runs
separately; they do not overwrite the reference CSVs or figures in this report.
