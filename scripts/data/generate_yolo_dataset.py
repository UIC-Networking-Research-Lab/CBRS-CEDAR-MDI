"""
CBRS dataset - spectrogram generation for YOLO training.

Converts every raw SigMF frame into a clean spectrogram image (no frame, title or
ticks) plus a YOLO label file.

Output layout (one folder per requested resolution):
    dataset_medium/  (640x640)   - standard YOLO input, used in the paper (default)
    dataset_large/   (1280x1280) - optional
    dataset_small/   (416x416)   - optional

Each dataset folder contains:
    images/     spectrogram PNG files
    labels/     YOLO-format label files
    metadata/   copy of the SigMF metadata with sample-rate corrections
    demo/       annotated preview images for human inspection (optional)

Dataset notes:
    - Sample rate: 30.72 MHz (the 184.32 MHz value in the SigMF metadata is wrong)
    - Frame length: 40 ms
    - Frequency span: +/-15.36 MHz around the 3600 MHz center frequency

Usage (from the repository root):
    python scripts/data/generate_yolo_dataset.py
    python scripts/data/generate_yolo_dataset.py --limit 20 --no-demo   # quick test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

# VERSION WARNING:
# Spectrogram PNG pixels are version-sensitive. Dataset generation, yolo_split
# PNG images, trained models, SigMF raw conversion, and inference must
# use the same rendering stack:
#   matplotlib==3.10.8
#   numpy==2.2.4
#   pillow==11.2.1
# This version set has passed virtual-PNG pixel validation against yolo_split.

import json
import argparse
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============== Configuration ==============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "IEEE_UPLOAD"
OUTPUT_BASE = PROJECT_ROOT

# Available resolutions (width x height). The paper uses 'medium' only.
RESOLUTIONS = {
    'large':  (1280, 1280),
    'medium': (640, 640),
    'small':  (416, 416),
}
DEFAULT_RESOLUTIONS = ['medium']

# Annotated demo image (for human inspection)
DEMO_FIGSIZE = (10, 6)  # inches
DEMO_DPI = 100          # -> 1000x600 pixels, about 200-500 KB per file

# Multiprocessing
NUM_WORKERS = 12

# Spectrogram parameters
NFFT = 512
NOVERLAP = 384

# Actual sample rate (the 184.32 MHz value in the metadata is wrong!)
ACTUAL_SAMPLE_RATE = 30.72e6  # 30.72 MHz

# Signal parameters
CENTER_FREQ = 3600e6  # 3600 MHz
RADAR_BANDWIDTH = 2e6  # 2 MHz

# ============== Data loading ==============

def load_sigmf_data(data_path):
    """Load a SigMF data file (cf32_le)."""
    return np.fromfile(data_path, dtype=np.complex64)

def load_sigmf_meta(meta_path):
    """Load a SigMF metadata file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============== Spectrogram ==============

def generate_spectrogram(iq_data, sample_rate, nfft=512, noverlap=384):
    """
    Compute the spectrogram.
    Returns: frequency array, time array, magnitude spectrum in dB.
    """
    f, t, Sxx = signal.spectrogram(
        iq_data,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        return_onesided=False,
        mode='magnitude'
    )

    # Move zero frequency to the center (fftshift)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f)

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    return f, t, Sxx_db

# ============== Label extraction ==============

def extract_labels(meta, folder_name):
    """
    Extract YOLO-format labels from the SigMF metadata.

    Returns: [(class_id, x_center, y_center, width, height), ...]
    with all coordinates normalized to [0, 1].

    Classes:
        0: LTE
        1: Radar
    """
    labels = []

    if 'annotations' not in meta or len(meta['annotations']) == 0:
        return labels

    annotation = meta['annotations'][0]

    # Spectrogram frequency span: [-15.36, +15.36] MHz
    freq_range = ACTUAL_SAMPLE_RATE / 1e6  # 30.72 MHz total
    freq_min = -freq_range / 2  # -15.36
    freq_max = freq_range / 2   # +15.36

    # LTE bandwidth from the folder name (5M/10M/15M/20M)
    lte_bw_from_folder = int(folder_name.replace('M', '')) * 1e6

    if 'system_components:transmitter' in annotation:
        transmitters = annotation['system_components:transmitter']

        # LTE signal (class_id = 0)
        if 'Tx_0' in transmitters:
            tx0 = transmitters['Tx_0']
            if 'signal:detail' in tx0 and 'signal:emitter' in tx0:
                detail = tx0['signal:detail']
                emitter = tx0['signal:emitter']

                lte_freq = float(emitter.get('frequency', 0))
                lte_bw = detail.get('bandwidth', 0)
                if lte_bw == 0:
                    lte_bw = lte_bw_from_folder

                # Offset from the center frequency (MHz)
                lte_offset = (lte_freq - CENTER_FREQ) / 1e6
                lte_bw_mhz = lte_bw / 1e6

                # Normalized coordinates (x axis = frequency)
                # x_center: relative position of the LTE center frequency
                x_center = (lte_offset - freq_min) / freq_range
                # width: LTE bandwidth as a fraction of the spectrogram width
                width = lte_bw_mhz / freq_range

                # y axis = time (LTE spans the whole frame)
                y_center = 0.5
                height = 1.0

                # Keep only valid boxes
                if 0 <= x_center <= 1 and width > 0:
                    labels.append((0, x_center, y_center, width, height))

        # Radar signal (class_id = 1)
        if 'Tx_1' in transmitters:
            tx1 = transmitters['Tx_1']
            if 'signal:detail' in tx1 and 'signal:emitter' in tx1:
                detail = tx1['signal:detail']
                emitter = tx1['signal:emitter']

                radar_freq = float(emitter.get('frequency', 0))
                radar_bw = detail.get('bandwidth', RADAR_BANDWIDTH)

                # Offset from the center frequency (MHz)
                radar_offset = (radar_freq - CENTER_FREQ) / 1e6
                radar_bw_mhz = radar_bw / 1e6

                # Normalized coordinates
                x_center = (radar_offset - freq_min) / freq_range
                width = radar_bw_mhz / freq_range

                # Radar also spans the whole time axis (pulses occur across the frame)
                y_center = 0.5
                height = 1.0

                if 0 <= x_center <= 1 and width > 0:
                    labels.append((1, x_center, y_center, width, height))

    return labels


def format_yolo_labels(labels):
    """Format labels in YOLO txt format."""
    lines = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def save_yolo_label(labels, output_path):
    """Write a YOLO label file."""
    with open(output_path, 'w') as f:
        f.write(format_yolo_labels(labels))


def save_demo_spectrogram(Sxx_db_T, Sxx_db, f, t, meta, folder_name, output_path):
    """
    Save an annotated demo spectrogram for human inspection, including the
    title, ground-truth signal boxes and the overlap scenario.
    """
    fig, ax = plt.subplots(figsize=DEMO_FIGSIZE, dpi=DEMO_DPI)

    # Spectrogram
    extent = [f[0]/1e6, f[-1]/1e6, t[-1]*1000, t[0]*1000]  # x = frequency (MHz), y = time (ms)
    im = ax.imshow(
        Sxx_db_T,
        aspect='auto',
        origin='upper',
        extent=extent,
        cmap='viridis',
        vmin=np.percentile(Sxx_db, 5),
        vmax=np.percentile(Sxx_db, 95)
    )

    # Signal information from the metadata
    annotation = meta.get('annotations', [{}])[0]
    transmitters = annotation.get('system_components:transmitter', {})

    # LTE bandwidth
    lte_bw_from_folder = int(folder_name.replace('M', '')) * 1e6

    # Frequency span
    freq_range = ACTUAL_SAMPLE_RATE / 1e6  # 30.72 MHz

    is_overlapping = False
    lte_freq_left = None
    lte_freq_right = None
    gt_signals = []
    sinr_value = None

    # SINR
    if 'system_components:receiver' in annotation:
        sinr_str = annotation['system_components:receiver'].get('SINR', None)
        if sinr_str:
            try:
                sinr_value = float(sinr_str)
            except:
                pass

    # LTE box (blue)
    if 'Tx_0' in transmitters:
        tx0 = transmitters['Tx_0']
        if 'signal:detail' in tx0 and 'signal:emitter' in tx0:
            detail = tx0['signal:detail']
            emitter = tx0['signal:emitter']

            lte_freq = float(emitter.get('frequency', 0))
            lte_bw = detail.get('bandwidth', 0)
            if lte_bw == 0:
                lte_bw = lte_bw_from_folder

            lte_offset_mhz = (lte_freq - CENTER_FREQ) / 1e6
            lte_freq_left = lte_offset_mhz - lte_bw/2/1e6
            lte_freq_right = lte_offset_mhz + lte_bw/2/1e6

            rect_lte = mpatches.Rectangle(
                (lte_freq_left, t[0]*1000),
                lte_freq_right - lte_freq_left,
                (t[-1] - t[0])*1000,
                linewidth=2.5,
                edgecolor='blue',
                facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect_lte)

            ax.text(
                lte_freq_left + 0.3,
                t[0]*1000 + 1,
                'LTE',
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='blue', alpha=0.9),
                verticalalignment='top'
            )
            gt_signals.append(f"LTE({lte_bw/1e6:.0f}MHz)")

    # Radar box (red)
    if 'Tx_1' in transmitters:
        tx1 = transmitters['Tx_1']
        if 'signal:detail' in tx1 and 'signal:emitter' in tx1:
            detail = tx1['signal:detail']
            emitter = tx1['signal:emitter']

            radar_freq = float(emitter.get('frequency', 0))
            radar_bw = detail.get('bandwidth', RADAR_BANDWIDTH)

            radar_offset_mhz = (radar_freq - CENTER_FREQ) / 1e6
            radar_freq_left = radar_offset_mhz - radar_bw/2/1e6
            radar_freq_right = radar_offset_mhz + radar_bw/2/1e6

            # Frequency overlap check
            if lte_freq_left is not None and lte_freq_right is not None:
                if radar_freq_left < lte_freq_right and radar_freq_right > lte_freq_left:
                    is_overlapping = True

            rect_radar = mpatches.Rectangle(
                (radar_freq_left, t[0]*1000),
                radar_freq_right - radar_freq_left,
                (t[-1] - t[0])*1000,
                linewidth=2.5,
                edgecolor='red',
                facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect_radar)

            ax.text(
                radar_freq_left + 0.1,
                t[0]*1000 + 1,
                'Radar',
                color='white',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='red', alpha=0.9),
                verticalalignment='top'
            )
            gt_signals.append("Radar(2MHz)")

    # Axis labels
    ax.set_xlabel('Frequency Offset (MHz)', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)

    # Title
    gt_text = ", ".join(gt_signals) if gt_signals else "None"
    title = f'CBRS Spectrogram - {folder_name} | Ground Truth: {gt_text}'
    if sinr_value is not None:
        title += f' | SINR: {sinr_value:.1f}dB'
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Overlap scenario tag
    scenario_text = "overlapping" if is_overlapping else "non overlapping"
    ax.text(
        0.95, 0.05,
        scenario_text,
        transform=ax.transAxes,
        fontsize=11,
        color='white',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.6)
    )

    # Colorbar
    plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DEMO_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ============== Dataset generation ==============

def get_all_files():
    """Collect all SigMF (meta, data) file pairs."""
    all_files = []
    folders = ['5M', '10M', '15M', '20M']

    for folder in folders:
        folder_path = BASE_DIR / folder
        if folder_path.exists():
            meta_files = sorted(folder_path.glob("*.sigmf-meta"))
            for meta_path in meta_files:
                data_path = meta_path.with_suffix('.sigmf-data')
                if data_path.exists():
                    all_files.append((folder, meta_path, data_path))

    return all_files

def build_expected_paths(base_name, dataset_dirs, generate_demo):
    """Return every output path that must exist for a completed sample."""
    expected_paths = []
    for dirs in dataset_dirs.values():
        expected_paths.extend([
            dirs['images'] / f"{base_name}.png",
            dirs['labels'] / f"{base_name}.txt",
            dirs['metadata'] / f"{base_name}.json",
        ])
    if generate_demo:
        demo_dirs = dataset_dirs.get('medium') or next(iter(dataset_dirs.values()))
        expected_paths.append(demo_dirs['demo'] / f"{base_name}_demo.png")
    return expected_paths


def process_dataset_file(file_info, dataset_dirs, generate_demo, overwrite):
    """
    Process one SigMF sample once, then save all requested resolution outputs
    into the shared dataset folders.
    """
    folder, meta_path, data_path = file_info
    base_name = meta_path.stem

    expected_paths = build_expected_paths(base_name, dataset_dirs, generate_demo)
    if not overwrite and all(path.exists() for path in expected_paths):
        return True, None

    try:
        iq_data = load_sigmf_data(str(data_path))
        meta = load_sigmf_meta(str(meta_path))

        f, t, Sxx_db = generate_spectrogram(iq_data, ACTUAL_SAMPLE_RATE, NFFT, NOVERLAP)
        Sxx_db_T = Sxx_db.T
        labels = extract_labels(meta, folder)

        for dirs in dataset_dirs.values():
            image_path = dirs['images'] / f"{base_name}.png"
            save_spectrogram_from_data(
                Sxx_db_T,
                Sxx_db,
                str(image_path),
                dirs['resolution'],
            )

        for dirs in dataset_dirs.values():
            label_path = dirs['labels'] / f"{base_name}.txt"
            save_yolo_label(labels, str(label_path))

        if generate_demo:
            demo_dirs = dataset_dirs.get('medium') or next(iter(dataset_dirs.values()))
            demo_path = demo_dirs['demo'] / f"{base_name}_demo.png"
            save_demo_spectrogram(Sxx_db_T, Sxx_db, f, t, meta, folder, str(demo_path))

        meta = add_metadata_corrections(meta, folder)

        for dirs in dataset_dirs.values():
            meta_output_path = dirs['metadata'] / f"{base_name}.json"
            with open(meta_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(meta, f_out, indent=2, ensure_ascii=False)

        return True, None
    except Exception as e:
        return False, f"{folder}/{base_name}: {e}"


def add_metadata_corrections(meta, folder):
    """Add the dataset-specific metadata corrections used by generated outputs."""
    meta['_actual_sample_rate'] = ACTUAL_SAMPLE_RATE
    meta['_lte_bandwidth_folder'] = folder
    meta['_note'] = "core:sample_rate in metadata is WRONG. Actual rate is 30.72 MHz."
    return meta


def images_match(path_a, path_b, tolerance=1.0 / 255.0 + 1e-6):
    """Compare two generated PNGs by decoded pixel values."""
    image_a = plt.imread(path_a)
    image_b = plt.imread(path_b)
    if image_a.shape != image_b.shape:
        return False, f"shape mismatch {image_a.shape} != {image_b.shape}"

    max_diff = float(np.max(np.abs(image_a.astype(np.float32) - image_b.astype(np.float32))))
    if max_diff > tolerance:
        return False, f"pixel diff too large {max_diff:.8f}"

    return True, None


def validate_dataset_outputs(all_files, dataset_dirs, generate_demo, validate_fraction=0.05, seed=42):
    """
    Validate that every expected output exists, then randomly recompute a fraction
    of samples and compare labels, metadata, and generated images.
    """
    missing = []
    for _, meta_path, _ in all_files:
        base_name = meta_path.stem
        for path in build_expected_paths(base_name, dataset_dirs, generate_demo):
            if not path.exists():
                missing.append(str(path))

    if missing:
        print(f"\nValidation failed: {len(missing)} output files are missing")
        for path in missing[:20]:
            print(f"  missing: {path}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return False

    if not all_files or validate_fraction <= 0:
        print("\nValidation done: all output files exist (sampled recomputation disabled)")
        return True

    sample_count = max(1, math.ceil(len(all_files) * validate_fraction))
    sample_count = min(sample_count, len(all_files))
    sampled_files = random.Random(seed).sample(all_files, sample_count)
    validation_dir = OUTPUT_BASE / "_validation_tmp"
    validation_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    print(f"\nSampled recomputation check: {sample_count}/{len(all_files)} ({validate_fraction:.1%})")

    for folder, meta_path, data_path in tqdm(sampled_files, desc="Validating"):
        base_name = meta_path.stem
        try:
            iq_data = load_sigmf_data(str(data_path))
            meta = load_sigmf_meta(str(meta_path))
            f, t, Sxx_db = generate_spectrogram(iq_data, ACTUAL_SAMPLE_RATE, NFFT, NOVERLAP)
            Sxx_db_T = Sxx_db.T
            labels = extract_labels(meta, folder)
            expected_label_text = format_yolo_labels(labels)
            expected_meta = add_metadata_corrections(meta, folder)

            for res_name, dirs in dataset_dirs.items():
                label_path = dirs['labels'] / f"{base_name}.txt"
                with open(label_path, 'r') as f_label:
                    actual_label_text = f_label.read()
                if actual_label_text != expected_label_text:
                    failures.append(f"{res_name}/{base_name}: label mismatch")

                meta_path_out = dirs['metadata'] / f"{base_name}.json"
                with open(meta_path_out, 'r', encoding='utf-8') as f_meta:
                    actual_meta = json.load(f_meta)
                if actual_meta != expected_meta:
                    failures.append(f"{res_name}/{base_name}: metadata mismatch")

                temp_image_path = validation_dir / f"{base_name}_{res_name}.png"
                save_spectrogram_from_data(
                    Sxx_db_T,
                    Sxx_db,
                    str(temp_image_path),
                    dirs['resolution'],
                )
                image_path = dirs['images'] / f"{base_name}.png"
                ok, reason = images_match(str(image_path), str(temp_image_path))
                if not ok:
                    failures.append(f"{res_name}/{base_name}: image mismatch ({reason})")
                temp_image_path.unlink(missing_ok=True)

            if generate_demo:
                demo_dirs = dataset_dirs.get('medium') or next(iter(dataset_dirs.values()))
                temp_demo_path = validation_dir / f"{base_name}_demo.png"
                save_demo_spectrogram(Sxx_db_T, Sxx_db, f, t, expected_meta, folder, str(temp_demo_path))
                demo_path = demo_dirs['demo'] / f"{base_name}_demo.png"
                ok, reason = images_match(str(demo_path), str(temp_demo_path))
                if not ok:
                    failures.append(f"demo/{base_name}: image mismatch ({reason})")
                temp_demo_path.unlink(missing_ok=True)

        except Exception as e:
            failures.append(f"{folder}/{base_name}: validation error {e}")

    try:
        validation_dir.rmdir()
    except OSError:
        pass

    if failures:
        print(f"\nValidation failed: {len(failures)} problems")
        for failure in failures[:20]:
            print(f"  {failure}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return False

    print("\nValidation done: all output files exist and sampled recomputation matches")
    return True


def create_all_datasets(
    active_res=None,
    generate_demo=True,
    limit=None,
    overwrite=False,
    workers=NUM_WORKERS,
    validate=True,
    validate_fraction=0.05,
    validation_seed=42,
):
    """
    Create the datasets for all requested resolutions in one pass.
    Each sample is read once and saved at every requested resolution, plus one demo image.
    """
    # Resolutions to generate
    if active_res is None:
        active_res = RESOLUTIONS

    # One directory tree per resolution
    dataset_dirs = {}
    for res_name, resolution in active_res.items():
        dataset_dir = OUTPUT_BASE / f"dataset_{res_name}"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        metadata_dir = dataset_dir / "metadata"
        demo_dir = dataset_dir / "demo"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        demo_dir.mkdir(parents=True, exist_ok=True)

        dataset_dirs[res_name] = {
            'images': images_dir,
            'labels': labels_dir,
            'metadata': metadata_dir,
            'demo': demo_dir,
            'resolution': resolution
        }

    # Input files
    all_files = get_all_files()
    if limit is not None:
        all_files = all_files[:limit]

    print(f"\n{'='*60}")
    print(f"Resolutions: {list(active_res.keys())} + demo images")
    print(f"Total files: {len(all_files)}")
    print(f"{'='*60}")

    worker_count = max(1, min(workers, len(all_files))) if all_files else 0
    print(f"Parallel workers: {worker_count} / requested {workers}")

    if not all_files:
        print(f"\nError: no SigMF file pairs found. Check the input directory: {BASE_DIR}")
        return 0, 1

    success = 0
    error_messages = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_dataset_file,
                file_info,
                dataset_dirs,
                generate_demo,
                overwrite,
            )
            for file_info in all_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
            ok, error = future.result()
            if ok:
                success += 1
            else:
                error_messages.append(error)

    for error in error_messages:
        print(f"\nError [{error}]")

    errors = len(error_messages)
    print(f"\nDone: {success} succeeded, {errors} failed")
    if validate:
        validation_ok = validate_dataset_outputs(
            all_files,
            dataset_dirs,
            generate_demo,
            validate_fraction=validate_fraction,
            seed=validation_seed,
        )
        if not validation_ok:
            errors += 1
    return success, errors

def save_spectrogram_from_data(Sxx_db_T, Sxx_db, output_path, resolution):
    """
    Save a spectrogram image from precomputed spectrum data (avoids recomputing the FFT).
    """
    dpi = 100
    width_inch = resolution[0] / dpi
    height_inch = resolution[1] / dpi

    fig = plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(
        Sxx_db_T,
        aspect='auto',
        origin='upper',
        cmap='viridis',
        vmin=np.percentile(Sxx_db, 5),
        vmax=np.percentile(Sxx_db, 95)
    )

    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='black'
    )
    plt.close(fig)


def create_dataset_yaml(resolution_name):
    """Create the YOLO dataset config file (before splitting)."""
    dataset_dir = OUTPUT_BASE / f"dataset_{resolution_name}"

    yaml_content = f"""# CBRS Dataset - {resolution_name}
# Auto-generated by generate_yolo_dataset.py. Run split_yolo_dataset.py to build
# the train/val/test split used for training.

path: {dataset_dir}
train: images
val: images
test:

# Classes
names:
  0: LTE
  1: Radar

# Dataset info
# Resolution: {RESOLUTIONS[resolution_name][0]}x{RESOLUTIONS[resolution_name][1]}
# Sample rate: 30.72 MHz (NOT 184.32 MHz as stated in metadata!)
# Frame duration: 40 ms
# Frequency range: +/-15.36 MHz (centered at 3600 MHz)
"""

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"Wrote dataset config: {yaml_path}")

def main():
    """Generate the datasets for the requested resolutions (one read per sample)."""
    global BASE_DIR, OUTPUT_BASE

    parser = argparse.ArgumentParser(description="Generate CBRS YOLO datasets")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=list(RESOLUTIONS.keys()),
        default=DEFAULT_RESOLUTIONS,
        help="Dataset sizes to generate. Default: medium (640x640, used in the paper)",
    )
    parser.add_argument("--no-demo", action="store_true", help="Skip annotated demo images for faster generation")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples for a quick test")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even if outputs already exist")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Parallel workers. Default: 12")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-generation validation")
    parser.add_argument("--validate-fraction", type=float, default=0.05, help="Fraction of samples to recompute for validation. Default: 0.05")
    parser.add_argument("--validation-seed", type=int, default=42, help="Random seed for validation sampling")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR, help="Raw dataset directory containing 5M/10M/15M/20M")
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE, help="Directory where dataset_* folders are written")
    args = parser.parse_args()
    BASE_DIR = args.base_dir.expanduser().resolve()
    OUTPUT_BASE = args.output_base.expanduser().resolve()
    active_res = {k: v for k, v in RESOLUTIONS.items() if k in args.resolutions}

    print("="*70)
    print("CBRS dataset - spectrogram generation for YOLO training")
    print("="*70)
    print(f"\nInput directory:  {BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE}")
    print("\nResolutions:")
    for name, res in active_res.items():
        print(f"  {name:8s}: {res[0]}x{res[1]}")
    skipped_res = [k for k in RESOLUTIONS.keys() if k not in args.resolutions]
    if skipped_res:
        print(f"\n  (skipped: {skipped_res})")
    if not args.no_demo:
        print("  + demo images: 1000x600 (annotated)")

    print("\nNote:")
    print("  - The sample rate in the SigMF metadata (184.32 MHz) is wrong.")
    print("  - Actual sample rate: 30.72 MHz")
    print("  - Check: 1,228,800 samples / 30.72 MHz = 40 ms frame length")

    success, errors = create_all_datasets(
        active_res=active_res,
        generate_demo=not args.no_demo,
        limit=args.limit,
        overwrite=args.overwrite,
        workers=args.workers,
        validate=not args.no_validate,
        validate_fraction=args.validate_fraction,
        validation_seed=args.validation_seed,
    )

    # One data.yaml per generated dataset
    if success == 0 and errors > 0:
        print("\nNo samples were generated; skipping data.yaml creation.")
        return

    for res_name in active_res.keys():
        create_dataset_yaml(res_name)

    print("\n" + "="*70)
    print("All done.")
    print(f"Result: {success} succeeded, {errors} failed")
    print("\nGenerated datasets:")
    for res_name, resolution in active_res.items():
        dataset_dir = OUTPUT_BASE / f"dataset_{res_name}"
        print(f"  {res_name:8s} ({resolution[0]}x{resolution[1]}): {dataset_dir}")
    if not args.no_demo:
        print(f"  {'demo':8s} (1000x600): dataset_medium/demo/")
    print("="*70)

if __name__ == "__main__":
    main()
