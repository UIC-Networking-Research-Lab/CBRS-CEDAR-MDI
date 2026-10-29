"""
Build and validate a raw SigMF split that mirrors yolo_split.

No CLI is used. Edit the globals below, then run:

    python scripts/data/build_raw_sigmf_split.py

Default output:

    raw_sigmf_split/
      train/
      validation/
      test/
      manifest.csv

Each output split contains flat SigMF pairs:

    raw_sigmf_split/train/<sample>.sigmf-meta
    raw_sigmf_split/train/<sample>.sigmf-data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


# =========================
# Global run configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source raw SigMF root. The source can contain bandwidth subfolders such as 5M/10M.
RAW_SIGMF_DIR = PROJECT_ROOT / "IEEE_UPLOAD"

# Existing YOLO split root.
YOLO_SPLIT_DIR = PROJECT_ROOT / "yolo_split"

# New raw split output root.
OUTPUT_DIR = PROJECT_ROOT / "raw_sigmf_split"

# No CLI. Choose one of:
#   "validate_inputs"      Validate RAW_SIGMF_DIR against YOLO_SPLIT_DIR only.
#   "build"                Validate inputs, build OUTPUT_DIR, validate output.
#   "validate_output"      Validate OUTPUT_DIR against YOLO_SPLIT_DIR only.
RUN_MODE = "build"

# YOLO split name -> output raw split name.
SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

# If True, raw directories must not contain extra complete SigMF pairs beyond yolo_split.
STRICT_NO_EXTRA_RAW = True

# Existing destination files are skipped when sizes match.
# If a destination file exists with a different size, set this True to replace it.
OVERWRITE_MISMATCHED_DEST = False

# In RUN_MODE="build", if both YOLO_SPLIT_DIR and OUTPUT_DIR already exist,
# validate OUTPUT_DIR against YOLO_SPLIT_DIR and do not copy or create files.
CHECK_EXISTING_OUTPUT_ONLY = True

# Copy method. "copy" is portable and independent. "hardlink" saves disk space but
# requires source and destination to be on the same filesystem.
COPY_METHOD = "copy"  # "copy" or "hardlink"

MANIFEST_NAME = "manifest.csv"


@dataclass(frozen=True)
class YoloRecord:
    yolo_split: str
    output_split: str
    stem: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class RawPair:
    stem: str
    meta_path: Path
    data_path: Path
    band: str


@dataclass(frozen=True)
class BuildRecord:
    yolo: YoloRecord
    raw: RawPair
    dest_meta: Path
    dest_data: Path


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        fail(f"{label} does not exist: {path}")
    if not path.is_dir():
        fail(f"{label} is not a directory: {path}")
    return path


def sigmf_stem(path: Path) -> str:
    return path.name.replace(".sigmf-meta", "").replace(".sigmf-data", "")


def load_yolo_records(yolo_split_dir: Path) -> list[YoloRecord]:
    yolo_split_dir = require_dir(yolo_split_dir, "YOLO split directory")
    records: list[YoloRecord] = []
    seen_stems: dict[str, str] = {}
    errors: list[str] = []

    for yolo_split, output_split in SPLIT_MAP.items():
        images_dir = yolo_split_dir / yolo_split / "images"
        labels_dir = yolo_split_dir / yolo_split / "labels"

        if not images_dir.exists():
            errors.append(f"missing YOLO images directory: {images_dir}")
            continue
        if not labels_dir.exists():
            errors.append(f"missing YOLO labels directory: {labels_dir}")
            continue

        image_paths = sorted(images_dir.glob("*.png"))
        if not image_paths:
            errors.append(f"no YOLO images found: {images_dir}")
            continue

        for image_path in image_paths:
            stem = image_path.stem
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                errors.append(f"missing YOLO label for image: {image_path}")

            previous_split = seen_stems.get(stem)
            if previous_split is not None:
                errors.append(
                    f"duplicate YOLO sample stem across splits: {stem} "
                    f"({previous_split}, {yolo_split})"
                )
            seen_stems[stem] = yolo_split

            records.append(
                YoloRecord(
                    yolo_split=yolo_split,
                    output_split=output_split,
                    stem=stem,
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve(),
                )
            )

    if errors:
        print_validation_errors("YOLO split validation failed", errors)
        sys.exit(1)

    return records


def collect_sigmf_files(raw_dir: Path, recursive: bool) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    pattern = "**/*" if recursive else "*"
    meta_files: dict[str, list[Path]] = {}
    data_files: dict[str, list[Path]] = {}

    for path in raw_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.name.endswith(".sigmf-meta"):
            meta_files.setdefault(sigmf_stem(path), []).append(path.resolve())
        elif path.name.endswith(".sigmf-data"):
            data_files.setdefault(sigmf_stem(path), []).append(path.resolve())

    return meta_files, data_files


def index_raw_pairs(raw_dir: Path, recursive: bool = True) -> tuple[dict[str, RawPair], list[str]]:
    raw_dir = require_dir(raw_dir, "Raw SigMF directory")
    meta_files, data_files = collect_sigmf_files(raw_dir, recursive=recursive)
    all_stems = sorted(set(meta_files) | set(data_files))

    pairs: dict[str, RawPair] = {}
    errors: list[str] = []

    for stem in all_stems:
        metas = meta_files.get(stem, [])
        datas = data_files.get(stem, [])

        if len(metas) != 1:
            errors.append(f"{stem}: expected 1 .sigmf-meta, found {len(metas)}")
        if len(datas) != 1:
            errors.append(f"{stem}: expected 1 .sigmf-data, found {len(datas)}")
        if len(metas) != 1 or len(datas) != 1:
            continue

        meta_path = metas[0]
        data_path = datas[0]
        band = meta_path.parent.name
        pairs[stem] = RawPair(stem=stem, meta_path=meta_path, data_path=data_path, band=band)

    return pairs, errors


def print_validation_errors(title: str, errors: list[str], max_lines: int = 50) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    for error in errors[:max_lines]:
        print(f"  - {error}")
    if len(errors) > max_lines:
        print(f"  ... {len(errors) - max_lines} more")
    print("=" * 80)


def validate_source_raw_against_yolo(
    raw_dir: Path,
    yolo_split_dir: Path,
    strict_no_extra_raw: bool = STRICT_NO_EXTRA_RAW,
) -> tuple[list[YoloRecord], dict[str, RawPair]]:
    """Validate unsplit/raw source directory against all YOLO split samples."""
    yolo_records = load_yolo_records(yolo_split_dir)
    raw_pairs, raw_errors = index_raw_pairs(raw_dir, recursive=True)

    errors = list(raw_errors)
    yolo_stems = {record.stem for record in yolo_records}
    raw_stems = set(raw_pairs)

    for record in yolo_records:
        if record.stem not in raw_pairs:
            errors.append(
                f"missing raw SigMF pair for YOLO sample {record.yolo_split}/{record.stem}"
            )

    if strict_no_extra_raw:
        for stem in sorted(raw_stems - yolo_stems):
            errors.append(f"extra raw SigMF pair not present in yolo_split: {stem}")

    if errors:
        print_validation_errors("Raw source <-> YOLO split validation failed", errors)
        sys.exit(1)

    print("Input validation passed.")
    print(f"  YOLO samples: {len(yolo_records)}")
    print(f"  Raw pairs:    {len(raw_pairs)}")
    return yolo_records, raw_pairs


def validate_output_raw_against_yolo(
    output_dir: Path,
    yolo_split_dir: Path,
    strict_no_extra_raw: bool = STRICT_NO_EXTRA_RAW,
) -> None:
    """Validate raw_sigmf_split/train|validation|test against YOLO split folders."""
    output_dir = require_dir(output_dir, "Output raw split directory")
    yolo_records = load_yolo_records(yolo_split_dir)
    errors: list[str] = []

    expected_by_split: dict[str, set[str]] = {}
    for record in yolo_records:
        expected_by_split.setdefault(record.output_split, set()).add(record.stem)

    for output_split, expected_stems in expected_by_split.items():
        split_dir = output_dir / output_split
        if not split_dir.exists():
            errors.append(f"missing raw output split directory: {split_dir}")
            continue

        raw_pairs, raw_errors = index_raw_pairs(split_dir, recursive=False)
        errors.extend(f"{output_split}: {error}" for error in raw_errors)

        raw_stems = set(raw_pairs)
        for stem in sorted(expected_stems - raw_stems):
            errors.append(f"{output_split}: missing raw pair for YOLO sample {stem}")

        if strict_no_extra_raw:
            for stem in sorted(raw_stems - expected_stems):
                errors.append(f"{output_split}: extra raw pair not present in matching YOLO split: {stem}")

    if errors:
        print_validation_errors("Output raw split <-> YOLO split validation failed", errors)
        sys.exit(1)

    print("Output validation passed.")
    for yolo_split, output_split in SPLIT_MAP.items():
        count = sum(1 for record in yolo_records if record.output_split == output_split)
        print(f"  {output_split}: {count} SigMF pairs")


def build_records(yolo_records: list[YoloRecord], raw_pairs: dict[str, RawPair]) -> list[BuildRecord]:
    records: list[BuildRecord] = []
    for yolo_record in yolo_records:
        raw_pair = raw_pairs[yolo_record.stem]
        split_dir = OUTPUT_DIR / yolo_record.output_split
        records.append(
            BuildRecord(
                yolo=yolo_record,
                raw=raw_pair,
                dest_meta=split_dir / raw_pair.meta_path.name,
                dest_data=split_dir / raw_pair.data_path.name,
            )
        )
    return records


def copy_or_link_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return "skipped"
        if not OVERWRITE_MISMATCHED_DEST:
            fail(
                "destination exists with a different size. "
                f"Set OVERWRITE_MISMATCHED_DEST=True to replace it: {destination}"
            )
        destination.unlink()

    if COPY_METHOD == "copy":
        shutil.copy2(source, destination)
    elif COPY_METHOD == "hardlink":
        destination.hardlink_to(source)
    else:
        fail(f"unsupported COPY_METHOD: {COPY_METHOD}")

    return "written"


def write_manifest(records: list[BuildRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "output_split",
        "yolo_split",
        "band",
        "stem",
        "yolo_image",
        "yolo_label",
        "source_meta",
        "source_data",
        "dest_meta",
        "dest_data",
    ]

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "output_split": record.yolo.output_split,
                    "yolo_split": record.yolo.yolo_split,
                    "band": record.raw.band,
                    "stem": record.yolo.stem,
                    "yolo_image": str(record.yolo.image_path),
                    "yolo_label": str(record.yolo.label_path),
                    "source_meta": str(record.raw.meta_path),
                    "source_data": str(record.raw.data_path),
                    "dest_meta": str(record.dest_meta.resolve()),
                    "dest_data": str(record.dest_data.resolve()),
                }
            )


def build_raw_sigmf_split() -> None:
    if CHECK_EXISTING_OUTPUT_ONLY and YOLO_SPLIT_DIR.exists() and OUTPUT_DIR.exists():
        print("Existing YOLO split and raw SigMF split found.")
        print("Checking existing raw_sigmf_split only; no files will be copied or rebuilt.")
        validate_output_raw_against_yolo(
            OUTPUT_DIR,
            YOLO_SPLIT_DIR,
            strict_no_extra_raw=STRICT_NO_EXTRA_RAW,
        )
        return

    yolo_records, raw_pairs = validate_source_raw_against_yolo(
        RAW_SIGMF_DIR,
        YOLO_SPLIT_DIR,
        strict_no_extra_raw=STRICT_NO_EXTRA_RAW,
    )
    records = build_records(yolo_records, raw_pairs)

    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        meta_status = copy_or_link_file(record.raw.meta_path, record.dest_meta)
        data_status = copy_or_link_file(record.raw.data_path, record.dest_data)

        written += int(meta_status == "written") + int(data_status == "written")
        skipped += int(meta_status == "skipped") + int(data_status == "skipped")

        if index % 250 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} SigMF pairs")

    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    write_manifest(records, manifest_path)

    print("Build complete.")
    print(f"  Output:   {OUTPUT_DIR.resolve()}")
    print(f"  Manifest: {manifest_path.resolve()}")
    print(f"  Written files: {written}")
    print(f"  Skipped files: {skipped}")

    validate_output_raw_against_yolo(
        OUTPUT_DIR,
        YOLO_SPLIT_DIR,
        strict_no_extra_raw=STRICT_NO_EXTRA_RAW,
    )


def main() -> None:
    if RUN_MODE == "validate_inputs":
        validate_source_raw_against_yolo(
            RAW_SIGMF_DIR,
            YOLO_SPLIT_DIR,
            strict_no_extra_raw=STRICT_NO_EXTRA_RAW,
        )
        return

    if RUN_MODE == "build":
        build_raw_sigmf_split()
        return

    if RUN_MODE == "validate_output":
        validate_output_raw_against_yolo(
            OUTPUT_DIR,
            YOLO_SPLIT_DIR,
            strict_no_extra_raw=STRICT_NO_EXTRA_RAW,
        )
        return

    fail(f"unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
