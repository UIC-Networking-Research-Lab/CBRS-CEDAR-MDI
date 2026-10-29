# Part 4 – CEDAR-MDI: distributed CEDAR-YOLO with optimized partitions

CEDAR-YOLO has no multi-scale neck, so its computation forms one ordered chain of stages:
the raw-I/Q preprocessing steps, then the backbone stages, SPPF and the head. Every cut
in this chain carries a single tensor. CEDAR-MDI splits the chain into four consecutive
groups on nodes A → B → C → D. It chooses the cut positions offline from measured costs
and can compress the tensor at each eligible cut.

<p align="center"><img src="figures/cedar_yolo_mdi_comparison.png" width="95%" alt="CEDAR-MDI vs local"></p>

First complete the [Jetson environment, asset and network setup in part 3](3_yolo_mdi.md#1-prepare-the-jetson-testbed).
The installer places the earlier CEDAR-YOLO MDI checkpoints in `trained_models_mdi/`
on every node. Use this edition for the local baseline too ([weights.md](weights.md)).
All reported CEDAR-MDI runs and their local speed baselines execute on Jetson
CPUs in FP32. PC training and CEDAR quality-sweep results belong to a separate
environment and checkpoint edition.

## 1. Offline partition optimization

The partition rankings used in the paper are included in [`configs/`](../configs):

| File | Content |
|---|---|
| `asym_jetson_stage_compute_payload_stats.{json,csv}` | per-stage compute time and boundary payload of each model on the Jetson CPU |
| `asym_jetson_link_bandwidth_stats.{json,csv}` | effective link rates measured per bandwidth environment |
| `asym_jetson_partition_rankings_nocompress.{json,csv}` | ranked partitions without compression, per model and rate |
| `asym_jetson_partition_rankings_compressed.{json,csv}` | ranked partitions for each compressor, per model and rate |
| `asym_module_boundary_partitions_jetson_*_best.json` | best partition of each setting as a single-partition config |

For every model, bandwidth environment and compressor, the optimizer enumerates every
feasible assignment of the stage chain to the four nodes. It estimates the pipeline
bottleneck τ, the largest per-node compute or per-link transfer time, from the
profiles. Candidates are ranked by τ, with the total compute plus transfer time as the
tie-breaker. The runtime then deploys the top K = 3 candidates.

This is the optimization in paper Section V-C, using the pipeline model from
Section III-A. For a fixed model, compressor and network environment, a policy
p defines the stage groups and their assignment along the node route. We solve

$$
\min_{p\in\mathcal{P}}\;\max\left\{
\max_{v\in\mathcal{V}}\sum_{s\in\mathcal{S}_v(p)}t_s,
\max_{\ell\in\mathcal{L}}\frac{b_\ell(p)r_\ell(p,q)}{B_\ell}
\right\}.
$$

Here, t is profiled stage compute time, b is uncompressed boundary size, r is
compressed/uncompressed size ratio, and B is measured effective link rate.
Uncompressed boundaries and the detection return use r = 1. Minimizing the
bottleneck maximizes estimated throughput; it does not minimize individual-frame
end-to-end delay. The reference stage chain has 165 feasible partitions per model
and bandwidth environment. Candidate policies are estimated offline, and only
the top three are deployed for each setting.

Stage times and payload sizes use medians from local profiles. Preliminary MDI
runs provide per-link effective rates and payload-specific compression ratios;
the pooled median ratio is used when a boundary lacks its own compression profile.
Eligible input/model tensors are compressed. Earlier preprocessing outputs and
the final detection return are sent uncompressed.

The supplied rankings are ready to use for the reference experiment. To regenerate
them offline from the supplied profiles:

```bash
python reproduce.py --task offline
```

By default this reuses the released stage/link profiles in `configs/`, regenerates all
rankings in `experiment_reruns/offline_profiles/`, and leaves the reference files intact.
It needs neither weights nor raw data. For a different testbed, first measure local
stage costs using the command below. Then run a fixed-partition distributed sweep
with all bandwidth environments and compression settings, including uncompressed
transfers. These runs supply both link rates and measured compression ratios.
Build new profiles and rankings from those measurement directories:

```bash
python scripts/mdi/build_cedar_partition_profiles.py --measurements \
    --local-baseline-dir experiment_reruns/cedar_local \
    --distributed-results-dir experiment_reruns/cedar_mdi \
    --output experiment_reruns/my_profiles
```

Missing measured links or compression ratios cause an error; configured token-bucket
rates are never substituted for missing measurements. For the first profiling sweep,
use `--partition_json configs/asym_module_boundary_partitions_jetson_nocompress_best.json`
with `cedar_mdi.py`, then rerank. Pass the regenerated ranking JSONs to the runtime with
`--partition_rankings_nocompress_json` and `--partition_rankings_compressed_json`.
The optimizer only estimates candidates offline; it does not deploy them.

## 2. Run on four devices

```bash
IPS="--ip_a 192.168.1.20 --ip_b 192.168.1.21 --ip_c 192.168.1.22 --ip_d 192.168.1.23"

# on B, C and D
python scripts/mdi/cedar_mdi.py --node B $IPS --device cpu
python scripts/mdi/cedar_mdi.py --node C $IPS --device cpu
python scripts/mdi/cedar_mdi.py --node D $IPS --device cpu

# on A: the paper sweep
#   3 models x (no compression + 6 compressors) x 3 environments x top-3 partitions
python scripts/mdi/cedar_mdi.py --node A $IPS --device cpu \
    --output_dir experiment_reruns/cedar_mdi
```

Set `IPS` in every terminal, run B/C/D on their respective devices, and check their
`/health` endpoints as in part 3. For an initial two-frame check, replace the A
command with:

```bash
python scripts/mdi/cedar_mdi.py --node A $IPS --device cpu \
    --model_name sig_yolo_v8n_f20t20_c100 --compressor none \
    --sweep_rates_mbps 150 --partition_top_k 1 --max_samples 2 \
    --output_dir experiment_reruns/cedar_network_check
```

Restart B/C/D after the check before running the full sweep; A shuts them down
when it finishes. The full sweep evaluates 189 configurations on the test split.
Do not include the short network-check outputs in the experiment comparison.

Measure the matching local baseline on one Jetson:

```bash
python scripts/cedar/local_cedar_stage_cost_test.py --models_dir trained_models_mdi \
    --sweep --sweep_devices cpu --run_location jetson \
    --output_dir experiment_reruns/cedar_local
```

The local run starts with raw I/Q and includes preprocessing, matching the
distributed pipeline. Evaluating the prepared PNGs measures a different workload.

Useful options (node A):

| Option | Meaning |
|---|---|
| `--model_name sig_yolo_v8x_f20t20_c100` | one model instead of all three |
| `--compressor` / `--compression_param` | as in part 3, e.g. `--compressor quantization --compression_param int8`, or `--compressor none` |
| `--sweep_rates_mbps` | token-bucket settings for high/medium/low environments (default `150,50,25`); not measured link capacities |
| `--partition_top_k` | number of ranked partitions to deploy per setting (default 3) |
| `--partition_json` | run one fixed partition file instead of the rankings |
| `--models_dir` | checkpoint directory on each node (default `trained_models_mdi`) |
| `--max_samples`, `--max_inflight`, `--output_dir` | as in part 3; default output `distributed_stage_sweep_results_asym/` |

Node A admits up to six frames into the pipeline at a time. Each worker uses separate
receive, compute and send queues, so different frames can occupy different stages at
the same time. Before each run, A sends the settings to the workers and waits until they
are ready; the timer starts after that.

Functional test on one machine:

```bash
python scripts/mdi/run_local_4node.py scripts/mdi/cedar_mdi.py \
    --model_name sig_yolo_v8n_f20t20_c100 --compressor quantization --compression_param int8 \
    --sweep_rates_mbps 150 --max_samples 10 --run_location local --output_dir mdi_local_test
```

## 3. Outputs and plots

Each run writes a per-frame CSV named
`distributedModule_FastSigmf_<model>_<location>_cpu_<compressor>_rank0<k>_cuts<a>_<b>_<c>_rate<R>mbps_samples<N>.csv`
(`cuts` are the three cut positions in the stage chain), plus a `_summary.csv` and a
`_quality.json` for the whole sweep.

Compare with the local baseline (run part 2's `local_cedar_stage_cost_test.py` with
`--models_dir trained_models_mdi` on one Jetson):

```bash
python scripts/plots/plot_cedar_mdi_vs_local.py \
    --baseline_dir experiment_reruns/cedar_local \
    --distributed_dir experiment_reruns/cedar_mdi \
    --output_dir experiment_reruns/cedar_plots
```

Check that each expected model/compressor/environment/rank combination produced
its summary and quality record, with no failed frames. Report `seconds_per_task`
or inverse throughput separately from end-to-end task delay; a pipelined run can
improve the former while increasing the latter. The included
[experiment report](../results/REPORT.md) shows the reference comparisons.

## 4. Reference results

Four Jetson Orin Nano nodes, CPU, full 848-frame test split, 63 runs per model
(7 compression settings × 3 bandwidth environments × 3 ranked partitions).
The table summarizes runs within 0.01 overlap F1 of local execution to exclude
policies with severe quality degradation. Figure 9 includes all policies:

| Model | Local time per frame (s) | Runs kept | Throughput vs. local | Task delay vs. local |
|---|---:|---:|---:|---:|
| CEDAR-YOLO-n | 1.054 | 57 / 63 | 1.17–1.19× | 1.29–2.09× |
| CEDAR-YOLO-s | 1.310 | 57 / 63 | 1.45–1.47× | 1.22–2.12× |
| CEDAR-YOLO-x | 3.455 | 45 / 63 | **1.38–3.27×** | 1.82–4.32× |

* Pipelining raises the rate of completed frames, but each frame still passes through
  four nodes, so the per-frame task delay grows. Throughput here is the number of frames
  divided by the total run time.
* Preprocessing dominates the n and s pipelines, so distributing their small models
  offers smaller gains than for x. INT8 at high bandwidth gives the largest
  speed-up (3.27×) while keeping the local overlap quality.
* Uncompressed runs and the hybrid compressor match the local overlap F1 in every
  environment and for every candidate. INT4 and top-k depend on where the cut is placed.

The highest-speed-up case is extra-large, INT8, high bandwidth, R3: 0.948 frames/s
versus 0.289 locally, with overlap F1 = 1.000 in both cases. Its maximum observed
frame delay is 7.1 s; across quality-retained policies, the maximum is 16.1 s.
The [full report](../results/REPORT.md#4-cedar-mdi) also compares transferred
payload and compute balance against original-YOLO module-boundary MDI.

Data: [`results/paper/cedar_yolo_mdi_comparison_data.csv`](../results/paper/cedar_yolo_mdi_comparison_data.csv),
[`results/paper/cedar_mdi_timing_summary.csv`](../results/paper/cedar_mdi_timing_summary.csv).
