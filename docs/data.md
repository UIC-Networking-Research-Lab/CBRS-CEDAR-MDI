# Data preparation

## Fastest reproduction route: supplied PNGs

For a quick CEDAR-YOLO quality test, use the prepared spectrograms in the
[CBRS-CEDAR-MDI-ASSETS release](https://github.com/UIC-Networking-Research-Lab/CBRS-CEDAR-MDI/releases/tag/assets).
Install them with:

```bash
python reproduce.py --assets ../CBRS-CEDAR-MDI-ASSETS --task install
```

It contains **all 5,640 PNGs and labels at 640 × 640**, already in the fixed
train/val/test split under `datasets/`. Use them directly for training and image-based
evaluation. No raw I/Q download or spectrogram regeneration is needed for these tasks.
The single `splits.csv` records the original recording names and exact split membership;
the installer places it at `splits/cbrs_split.csv`. Original I/Q recordings are not
redistributed. CBRS-CEDAR-RT has its own independent download package.

**Changed preprocessing parameters or library versions can invalidate pretrained models.
For a different preprocessing pipeline, retrain your own model.** See
[the complete compatibility guide](preprocessing.md).

The remaining steps are for regenerating PNGs or reproducing raw-I/Q timing.

This guide turns the raw over-the-air recordings into the two inputs used by
every experiment:

| Folder | Content | Used by |
|---|---|---|
| `yolo_split/` | 640x640 spectrogram PNGs + YOLO labels, split into `train/`, `val/`, `test/` | training and PNG-based evaluation |
| `raw_sigmf_split/` | the same frames as raw SigMF pairs, split into `train/`, `validation/`, `test/` | end-to-end raw-I/Q inference (local and MDI) |

## 1. Download the raw dataset

**Only needed for raw-I/Q inputs, spectrogram regeneration or full-pipeline
preprocessing/timing tests.** The original Genesys recordings are not partitioned
into our training, validation and test sets. Use
[`splits/cbrs_split.csv`](../splits/cbrs_split.csv), identical to the assets'
`splits.csv`, to restore the exact membership. The supplied PNGs are already split.
For raw-frame restoration, follow [step 5](#5-build-the-raw-sigmf-split);
for regenerated PNGs, follow [step 4](#4-build-the-trainvaltest-split).

Download *"Real-world Radar and LTE Signals Dataset Collected Over-the-air in
Shared CBRS Band"* from <https://genesys-lab.org/CBRS>. Place the four LTE-bandwidth
folders under `IEEE_UPLOAD/` in the repository root:

```
IEEE_UPLOAD/
  5M/   rx_data_record_<timestamp>.sigmf-meta + .sigmf-data
  10M/
  15M/
  20M/
```

The dataset has 5,640 frames. Each frame holds 40 ms of complex I/Q samples
(cf32) at 30.72 MS/s, centered at 3.6 GHz. The LTE bandwidth is 5/10/15/20 MHz,
the radar bandwidth is 2 MHz, and the SINR ranges from 15 to 35 dB.

> The `core:sample_rate` field in the SigMF metadata (184.32 MHz) is wrong. The
> actual rate is 30.72 MHz: 1,228,800 samples / 30.72 MHz = 40 ms. The scripts use the correct value.

You can pass a different location with `--base-dir` (generator) or by editing
`RAW_SIGMF_DIR` (raw split builder).

## 2. Preserve the rendering environment

Spectrogram pixels depend on library versions and parameters. Use the supplied
PNGs directly for training and quality evaluation. If regenerating them, the
tested PC rendering profile is:

```
numpy==2.2.4  scipy==1.14.1  matplotlib==3.10.8  pillow==11.2.1
```

Jetson local timing and MDI use SciPy **1.17.1**, with the same NumPy, Matplotlib
and Pillow versions. `requirements.txt` selects SciPy by platform. These are
separate tested rendering profiles; other inference/training libraries need not
all match exact versions. Rendering changes can shift the model input distribution
even when the images look similar, making the pretrained weights unreliable.
Keep rendering settings fixed and run the [pixel comparison](preprocessing.md)
before using regenerated PNGs or raw-I/Q inference. Retrain on your own consistently
rendered dataset if you change preprocessing.

## 3. Generate the spectrogram dataset

```bash
python scripts/data/generate_yolo_dataset.py --no-demo
```

Output: `dataset_medium/{images,labels,metadata}/` with one PNG and one label file per frame.

* Spectrogram: 512-point FFT with 384-sample overlap, `fftshift`, `10*log10(|S|+1e-10)`,
  color limits at the 5th/95th percentile of each frame, `viridis` colormap.
  Frequency runs along the x axis and time along the y axis.
* Labels: class `0` = LTE and `1` = radar. Each box spans the signal's occupied
  frequency interval and the whole 40-ms frame, taken from the SigMF annotations.
* `--limit 20` processes only 20 frames for a quick test. `--workers` sets the
  number of processes. Drop `--no-demo` to also write annotated preview images.
  `--resolutions large small` adds 1280/416-pixel versions, which the paper does not use.
* The script ends with a validation pass. It checks that every output exists,
  recomputes 5 % of the frames and compares them with the written files.

Regenerated PNGs match the images used to train the released models. In our
check, 7 of 8 frames were bit-identical and the eighth differed by one intensity
level in 26 pixels, which is within the generator's own tolerance.

## 4. Build the train/val/test split

```bash
python scripts/data/split_yolo_dataset.py
```

The script copies `dataset_medium/` into `yolo_split/{train,val,test}/{images,labels}`
and writes `yolo_split/data.yaml`. By default it follows
[`splits/cbrs_split.csv`](../splits/cbrs_split.csv), the fixed split used for
every model and result in the paper:

| Split | Frames |
|---|---:|
| train | 3,946 |
| val | 846 |
| test | 848 (558 overlapping, 290 non-overlapping) |

Within each LTE-bandwidth group, the frames were split 70/15/15 at random.
Keep `USE_SPLIT_MANIFEST = True` when reproducing the paper. A new random split
changes the evaluation set and is a different experiment.

## 5. Build the raw SigMF split

```bash
python reproduce.py --task raw-split --raw-source /path/to/IEEE_UPLOAD
```

This copies the SigMF pairs from `IEEE_UPLOAD/` into `raw_sigmf_split/{train,validation,test}/`
so that they mirror `yolo_split/` exactly, using the exact sample membership in `splits/cbrs_split.csv`.
The raw-I/Q inference scripts in parts 1–4 read frames from here and labels from `yolo_split/`.
Use `python scripts/data/restore_raw_split.py --source /path/to/IEEE_UPLOAD --method hardlink`
on the same filesystem to avoid duplicating the raw bytes (about 9.8 MB per frame).

## Disk space

| Item | Size |
|---|---:|
| `IEEE_UPLOAD/` | ~55 GB |
| `raw_sigmf_split/` | ~55 GB (copy) or ~0 (hard links) |
| `dataset_medium/` + `yolo_split/` | ~5 GB each |
