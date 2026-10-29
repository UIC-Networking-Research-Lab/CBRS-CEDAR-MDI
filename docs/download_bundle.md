# CBRS-CEDAR-MDI assets

Data and pretrained models for the **CBRS-CEDAR-MDI paper reproduction repository**.
This download is independent of the RT deployment project's asset package.

## Download and extract

**[Download CBRS-CEDAR-MDI-ASSETS v1.0 from GitHub Releases](https://github.com/UIC-Networking-Research-Lab/CBRS-CEDAR-MDI/releases/tag/assets)**

1. Open the release's **Assets** list and download **all 9 archive parts**:
   `CBRS-CEDAR-MDI-ASSETS.7z.001` through `CBRS-CEDAR-MDI-ASSETS.7z.009`.
   Keep all parts in the same folder with their original names.
2. Use **7-Zip** to extract `CBRS-CEDAR-MDI-ASSETS.7z.001` only. It reads the remaining
   parts automatically; do not extract each part separately.
3. Place the extracted `CBRS-CEDAR-MDI-ASSETS` folder beside the repository:

```text
workspace/
  CBRS-CEDAR-MDI/
  CBRS-CEDAR-MDI-ASSETS/
    datasets/
    weights/
    splits.csv
    README.md
```

Alternatively, extract from the folder containing all parts with:

```bash
7z x CBRS-CEDAR-MDI-ASSETS.7z.001
```

GitHub's **Source code (zip/tar.gz)** downloads contain code, not the datasets or
weights. Use this project's asset package; the MDI and RT packages are independent.

## Package contents

```text
CBRS-CEDAR-MDI-ASSETS/
  datasets/
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  weights/
    original_yolo/<model>/weights/best.pt
    cedar_jetson/<model>/weights/best.pt
    cedar_local_latest/<model>/weights/best.pt
  splits.csv
  README.md
```

## Use the data and models

Extract this folder beside the CBRS-CEDAR-MDI repository. From the repository root:

```bash
python reproduce.py --assets ../CBRS-CEDAR-MDI-ASSETS --task install
python reproduce.py --task check --weights jetson
python reproduce.py --task png --weights jetson
```

`datasets/` contains the prepared **640 × 640 PNGs and YOLO labels**: 3,946 training,
846 validation and 848 test samples. Use these PNGs directly to reproduce image-based
experiments. Class 0 is LTE; class 1 is radar. `splits.csv` records each sample's
split, original bandwidth group and recording name.

The installer places data in `yolo_split/` and keeps the model editions separate:

| Weight folder | Contents and use |
|---|---|
| `original_yolo/` | Nine original YOLO checkpoints and one RT-DETR-L checkpoint |
| `cedar_jetson/` | Three earlier CEDAR checkpoints for the paper's Jetson MDI experiments and their matching local baselines; installed in `trained_models_mdi/` |
| `cedar_local_latest/` | All 108 PC quality/architecture-sweep checkpoints from `trained_models_cedar_best`; installed in `trained_models/` |

Use the same checkpoint edition when comparing local and MDI execution.
To install and evaluate the PC checkpoint edition, run this from the MDI repository:

```bash
python reproduce.py --assets ../CBRS-CEDAR-MDI-ASSETS --task install
python reproduce.py --task check --weights latest
python reproduce.py --task png --weights latest --device 0
```

The nano and small 20×20, 100%-coverage PC checkpoints differ from the earlier
Jetson edition; the extra-large file is identical across those two editions.
These PC sweep weights belong only to the MDI asset package, not RT.

## Raw I/Q and preprocessing

For a quick CEDAR-YOLO test, use the supplied PNGs directly. Raw I/Q recordings
are needed only for raw-frame input, spectrogram regeneration or end-to-end tests
starting from preprocessing. Download the over-the-air radar/LTE dataset
from [Genesys Lab](https://genesys-lab.org/CBRS). It contains overlapping and
non-overlapping signals. Follow its terms and cite the dataset paper. Restore the
paper's fixed raw-data split using the repository command below. **The original
recordings have no train/validation/test partition matching our experiments.**
The command uses `splits/cbrs_split.csv`, supplied as `splits.csv` in the assets;
do not replace it with a new random split:

```bash
python reproduce.py --task raw-split --raw-source /path/to/IEEE_UPLOAD
```

**Preprocessing parameters and library versions change the spectrogram pixels and
can make pretrained models unreliable.** Use the supplied PNGs for reproduction.
For your own preprocessing pipeline, retrain a model on its output.
PC training and the CEDAR quality sweep use Conda `torch270_py312_cuda`;
local speed baselines and MDI run on Jetson in `/opt/venv`. The tested renderer
uses NumPy 2.2.4, Matplotlib 3.10.8 and Pillow 11.2.1, with SciPy 1.14.1 on the
PC and 1.17.1 on Jetson. Preserve both rendering versions and parameters:
even a small mismatch can invalidate pretrained predictions. Complete settings
and the pixel check are in the repository's `docs/preprocessing.md`.

The repository contains the scripts, installation metadata and experiment reports.
This download contains only datasets, weights, the split list and this README.
