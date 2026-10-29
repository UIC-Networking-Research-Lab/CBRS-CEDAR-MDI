# Pretrained checkpoints: choose the experiment edition

The **CBRS-CEDAR-MDI-ASSETS** download stores each checkpoint edition separately.
Install it with `python reproduce.py --assets ../CBRS-CEDAR-MDI-ASSETS --task install`.
Download all 9 parts from the [assets v1.0 release](https://github.com/UIC-Networking-Research-Lab/CBRS-CEDAR-MDI/releases/tag/assets) and
follow the [extraction instructions](download_bundle.md#download-and-extract).

| Bundle directory | Installed MDI path | Intended experiments |
|---|---|---|
| `weights/original_yolo/` | `trained_models/cbrs_*/weights/best.pt` | Original YOLO training-derived evaluation and original-YOLO MDI; RT also uses these nine YOLO weights |
| `weights/cedar_jetson/` | `trained_models_mdi/sig_yolo_*/weights/best.pt` | Paper Jetson CEDAR-MDI **and its matching local baseline**; also the published RT GPU deployment results |
| `weights/cedar_local_latest/` | `trained_models/sig_yolo_*/weights/best.pt` | Later PC PNG quality evaluation and architecture/feature-map sweep |

**Current bundle:** original YOLO, the Jetson-era CEDAR edition, and all **108 PC
quality/architecture-sweep checkpoints** are included. The latter were supplied in
`trained_models_cedar_best` and are packaged under `weights/cedar_local_latest/`.
Every checkpoint matches the architecture, parameter count and selected epoch
in the paper's sweep records. Install the bundle from the MDI repository root:

```bash
python reproduce.py --assets ../CBRS-CEDAR-MDI-ASSETS --task install
python reproduce.py --task check --weights latest
python reproduce.py --task png --weights latest --device 0
```

The installer keeps PC quality checkpoints in `trained_models/` and Jetson
checkpoints in `trained_models_mdi/`. Publish the updated repository together
with this asset package so a fresh checkout recognizes the complete weight set.
The download contains only datasets, weights, `splits.csv` and a README.

The three 20×20, 100%-coverage models were also re-evaluated on the full test
split on the PC GPU with the training software versions. Their mAP@0.5 and overlap F1
exactly match the corresponding recorded sweep values. The other configurations
passed checkpoint/architecture checks; their full quality sweep was not rerun.

## Why the editions must stay separate

Both editions were trained on the PC. The Jetson experiments used an earlier
CEDAR-YOLO training run; the later architecture/quality sweep also ran on the PC.
The `cedar_jetson` name identifies the edition deployed on Jetson, not where it was
trained. Matching model names and architecture do not imply matching
weights. A fair local-versus-MDI comparison uses the **same checkpoint on both sides**.

For the 20×20, 100%-coverage models, the PC nano and small checkpoint files differ
from their Jetson counterparts; the extra-large checkpoint is byte-identical.
Both edition directories retain that extra-large file so each intended workflow
remains complete. The PC sweep weights are not added to the RT package.

The following quality measurements illustrate the distinction (test split):

| Model | Jetson edition mAP@0.5 | Jetson overlap F1 | Later local edition mAP@0.5 | Later local overlap F1 |
|---|---:|---:|---:|---:|
| CEDAR-YOLO-n | 0.895 | 0.995 | 0.995 | 0.999 |
| CEDAR-YOLO-s | 0.995 | 0.999 | 0.995 | 0.996 |
| CEDAR-YOLO-x | 0.995 | 1.000 | 0.995 | 1.000 |

Overlap F1 is close across editions; the nano model's box-detection mAP differs more.
Neither edition is described as uniformly better on every metric. RT evaluates the
Jetson-edition checkpoints on the PC GPU using validation images and reports box-detection macro F1,
which is a different metric from overlap F1.

Sources: [local comparison](../results/paper/cedar_yolo_local_comparison_data.csv),
[Jetson local/MDI comparison](../results/paper/cedar_yolo_mdi_comparison_data.csv),
and [CPU results](../results/paper/cedar_yolo_cpu_results.csv).

## Verify identity

[release_checkpoints.json](../configs/release_checkpoints.json) associates every available
checkpoint with its edition, destination and SHA-256. [SHA256SUMS](../SHA256SUMS) lists
all available weights. `configs/release_assets.json` in this repository records the
checksums used by the installer; the download package needs no JSON or checksum files.

```bash
python reproduce.py --task check --weights jetson
python reproduce.py --task check --weights latest
```

The second command fails explicitly until the latest edition is installed. It never
falls back to `trained_models_mdi/`. Keep CEDAR run names unchanged, because the model
loaders read the feature-grid size and regression coverage from those names.

## Formats and retraining

- Original YOLO/RT-DETR: standard Ultralytics checkpoints; use Ultralytics 8.4.46.
- CEDAR-YOLO: a dictionary with model/EMA state dicts and architecture metadata; the
  supplied evaluator and `scripts/predict.py` build the matching network.
- New training writes to **`trained_models_retrained/`**. It does not overwrite the
  published checkpoint editions. The CEDAR loop records early stopping and resume state.
- The original YOLO scripts initialize from the official Ultralytics weights, downloading
  them on first use if absent. For offline retraining, place those initialization `.pt`
  files in the repository root beforehand. They are distinct from our CBRS-trained weights.

**Retrain your own model when preprocessing changes.** Different rendering-library
versions or spectrogram parameters can invalidate the released model's input assumptions.
Use the supplied PNGs directly for reproduction. See [preprocessing.md](preprocessing.md).

Original-YOLO/RT-DETR epoch logs and training arguments are preserved in
[results/training_logs/](../results/training_logs/). The released original models used
25 epochs; CEDAR uses a 100-epoch budget with early stopping, not a mandatory 100 epochs.
