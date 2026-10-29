# CBRS-CEDAR-MDI assets

Data and pretrained models for the **CBRS-CEDAR-MDI paper reproduction repository**.
This download is independent of the RT deployment project's asset package.

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

Raw I/Q recordings are not included. Download the over-the-air radar/LTE dataset
from [Genesys Lab](https://genesys-lab.org/CBRS). It contains overlapping and
non-overlapping signals. Follow its terms and cite the dataset paper. Restore the
paper's fixed raw-data split using the repository command:

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
