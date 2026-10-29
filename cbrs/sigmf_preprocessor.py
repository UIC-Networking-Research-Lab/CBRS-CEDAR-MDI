"""
In-memory SigMF raw IQ preprocessing for model inference.

The core output is a virtual PNG: PNG bytes produced in memory with the same
matplotlib rendering path used by generate_yolo_dataset.py. This mirrors the
training dataset artifact, where Ultralytics reads yolo_split/images/*.png from
disk. The inference path is also PNG/image oriented, so this module keeps the
preprocessed raw SigMF sample in PNG form for compatibility while avoiding
temporary files on disk.

The RGB array helpers are adapters only: they decode the virtual PNG from memory
so tests or inference wrappers can pass a numpy image to APIs that accept arrays.

Version warning:
    The trained model, yolo_split PNG generation, SigMF raw-file conversion, and
    inference code must use the same rendering stack. Pixel output is
    sensitive to matplotlib/numpy/Pillow versions. The currently validated
    versions are:

        matplotlib==3.10.8
        numpy==2.2.4
        pillow==11.2.1

See VERSION_WARNING.md for the project-wide warning. The same versions should
be used for dataset generation, yolo_split PNGs, trained models, SigMF raw-file
conversion, and inference.

Usage:

    from cbrs.sigmf_preprocessor import SigMFSpectrogramPreprocessor

    preprocessor = SigMFSpectrogramPreprocessor()
    png_bytes = preprocessor.raw_to_png_bytes(meta_path, data_path)
    image = preprocessor.raw_to_image_array(meta_path, data_path)
    # image.shape == (640, 640, 3), image.dtype == np.uint8

Direct self-test, controlled by variables inside main(), compares decoded pixels
from generated virtual PNGs against the matching yolo_split true PNGs:

    python cbrs/sigmf_preprocessor.py
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


ACTUAL_SAMPLE_RATE = 30.72e6
DEFAULT_NFFT = 512
DEFAULT_NOVERLAP = 384
DEFAULT_IMAGE_SIZE = (640, 640)  # width, height
DEFAULT_PERCENTILE_CLIP = (5.0, 95.0)
DEFAULT_DPI = 100
DEFAULT_CMAP = "viridis"


class SigMFSpectrogramPreprocessor:
    """
    Convert SigMF cf32_le raw IQ data into a training-compatible virtual PNG.

    Prefer raw_to_png_bytes() or raw_to_png_buffer() when the downstream code can
    consume file-like image data. Use raw_to_image_array() only as an adapter for
    APIs that accept decoded RGB numpy arrays.

    Processing details retained from generate_yolo_dataset.py:
    - Use corrected sample rate 30.72 MHz, not metadata sample_rate.
    - scipy.signal.spectrogram(..., return_onesided=False, mode="magnitude").
    - FFT-shift frequency bins before dB conversion.
    - Render Sxx_db.T so orientation matches generated training images.
    - Use per-sample 5/95 percentile clipping.
    - Use matplotlib Agg, viridis, borderless axes, bbox_inches="tight",
      pad_inches=0, and facecolor="black".
    """

    def __init__(
        self,
        sample_rate: float = ACTUAL_SAMPLE_RATE,
        nfft: int = DEFAULT_NFFT,
        noverlap: int = DEFAULT_NOVERLAP,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        percentile_clip: tuple[float, float] = DEFAULT_PERCENTILE_CLIP,
        dpi: int = DEFAULT_DPI,
        cmap: str = DEFAULT_CMAP,
    ) -> None:
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.noverlap = noverlap
        self.image_size = image_size
        self.percentile_clip = percentile_clip
        self.dpi = dpi
        self.cmap = cmap

    def raw_to_png_bytes(self, meta_path: str | Path, data_path: str | Path) -> bytes:
        """
        Convert one SigMF pair to PNG bytes in memory.

        meta_path is accepted to keep call sites explicit about the SigMF pair,
        but this CBRS preprocessing path intentionally ignores metadata because
        its sample_rate field is known to be wrong.
        """
        _ = meta_path
        return self.data_file_to_png_bytes(data_path)

    def raw_to_png_buffer(self, meta_path: str | Path, data_path: str | Path) -> io.BytesIO:
        """Convert one SigMF pair to a seekable in-memory PNG buffer."""
        return io.BytesIO(self.raw_to_png_bytes(meta_path, data_path))

    def raw_to_image_array(self, meta_path: str | Path, data_path: str | Path) -> np.ndarray:
        """
        Convert one SigMF pair to decoded 640x640x3 uint8 RGB from virtual PNG.

        This intentionally decodes the generated PNG bytes instead of bypassing
        PNG rendering, keeping the array consistent with the training PNG pixels.
        """
        return self._png_bytes_to_rgb_array(self.raw_to_png_bytes(meta_path, data_path))

    def data_file_to_png_bytes(self, data_path: str | Path) -> bytes:
        """Load a .sigmf-data file and convert it to virtual PNG bytes."""
        iq_data = np.fromfile(Path(data_path), dtype=np.complex64)
        return self.iq_to_png_bytes(iq_data)

    def iq_to_png_bytes(self, iq_data: np.ndarray) -> bytes:
        """Convert already-loaded IQ samples to virtual PNG bytes."""
        sxx_db = self._iq_to_spectrogram_db(iq_data)
        return self._spectrogram_db_to_png_bytes(sxx_db)

    def iq_to_image_array(self, iq_data: np.ndarray) -> np.ndarray:
        """Convert already-loaded IQ samples to decoded RGB from virtual PNG."""
        return self._png_bytes_to_rgb_array(self.iq_to_png_bytes(iq_data))

    def _iq_to_spectrogram_db(self, iq_data: np.ndarray) -> np.ndarray:
        """Generate a shifted dB spectrogram from in-memory IQ samples."""
        _frequencies, _times, sxx = signal.spectrogram(
            iq_data,
            fs=self.sample_rate,
            nperseg=self.nfft,
            noverlap=self.noverlap,
            return_onesided=False,
            mode="magnitude",
        )

        sxx = np.fft.fftshift(sxx, axes=0)
        return 10 * np.log10(sxx + 1e-10)

    def _spectrogram_db_to_png_bytes(self, sxx_db: np.ndarray) -> bytes:
        """
        Render shifted dB spectrogram data to PNG bytes in memory.

        This is the in-memory equivalent of generate_yolo_dataset.py's
        save_spectrogram_from_data(), with output_path replaced by BytesIO.
        """
        width_inch = self.image_size[0] / self.dpi
        height_inch = self.image_size[1] / self.dpi
        low, high = self.percentile_clip
        vmin = np.percentile(sxx_db, low)
        vmax = np.percentile(sxx_db, high)

        buffer = io.BytesIO()
        fig = plt.figure(figsize=(width_inch, height_inch), dpi=self.dpi)
        try:
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(
                sxx_db.T,
                aspect="auto",
                origin="upper",
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

            plt.savefig(
                buffer,
                format="png",
                dpi=self.dpi,
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
            )
        finally:
            plt.close(fig)

        return buffer.getvalue()

    @staticmethod
    def _png_bytes_to_rgb_array(png_bytes: bytes) -> np.ndarray:
        """Decode in-memory PNG bytes to HxWx3 uint8 RGB."""
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as image:
            return np.asarray(image.convert("RGB"))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # No CLI. Edit these local variables, then run this file directly.
    run_mode = "validate_random_raw_split"  # "validate_random_raw_split" or "none"
    raw_split_dir = project_root / "raw_sigmf_split"
    yolo_split_dir = project_root / "yolo_split"
    validation_sample_count = 25
    validation_seed = 42
    validation_pixel_tolerance = 1

    def format_errors(title: str, errors: list[str], max_lines: int = 50) -> str:
        lines = ["=" * 80, title, "=" * 80]
        lines.extend(f"  - {error}" for error in errors[:max_lines])
        if len(errors) > max_lines:
            lines.append(f"  ... {len(errors) - max_lines} more")
        lines.append("=" * 80)
        return "\n".join(lines)

    def expected_yolo_image_path(yolo_split_dir: Path, raw_split: str, stem: str) -> Path:
        yolo_split = "val" if raw_split == "validation" else raw_split
        return yolo_split_dir / yolo_split / "images" / f"{stem}.png"

    def collect_samples(
        raw_split_dir: Path,
        yolo_split_dir: Path,
    ) -> list[tuple[str, str, Path, Path, Path]]:
        raw_split_dir = raw_split_dir.expanduser().resolve()
        yolo_split_dir = yolo_split_dir.expanduser().resolve()
        if not raw_split_dir.exists():
            raise FileNotFoundError(f"Raw split directory not found: {raw_split_dir}")
        if not yolo_split_dir.exists():
            raise FileNotFoundError(f"YOLO split directory not found: {yolo_split_dir}")

        samples: list[tuple[str, str, Path, Path, Path]] = []
        errors: list[str] = []
        for split in ("train", "validation", "test"):
            split_dir = raw_split_dir / split
            if not split_dir.exists():
                errors.append(f"missing split directory: {split_dir}")
                continue

            for meta_path in sorted(split_dir.glob("*.sigmf-meta")):
                stem = meta_path.name.replace(".sigmf-meta", "")
                data_path = split_dir / f"{stem}.sigmf-data"
                yolo_image_path = expected_yolo_image_path(yolo_split_dir, split, stem)
                if not data_path.exists():
                    errors.append(f"missing data file: {data_path}")
                    continue
                if not yolo_image_path.exists():
                    errors.append(f"missing matching YOLO image path: {yolo_image_path}")
                    continue
                samples.append(
                    (
                        split,
                        stem,
                        meta_path.resolve(),
                        data_path.resolve(),
                        yolo_image_path.resolve(),
                    )
                )

        if errors:
            raise RuntimeError(format_errors("Raw split collection failed", errors))
        if not samples:
            raise RuntimeError("No raw split samples found.")
        return samples

    def read_reference_rgb(path: Path) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def validate_random_raw_split_samples() -> bool:
        """
        Randomly validate raw -> virtual PNG pixels against yolo_split true PNGs.

        The test compares decoded RGB pixels only. It does not compare raw PNG
        bytes, because PNG byte streams can differ even when decoded pixels match.
        """
        samples = collect_samples(raw_split_dir, yolo_split_dir)
        sample_count = min(validation_sample_count, len(samples))
        selected = random.Random(validation_seed).sample(samples, sample_count)
        preprocessor = SigMFSpectrogramPreprocessor()
        png_signature = b"\x89PNG\r\n\x1a\n"

        failures: list[str] = []
        print("=" * 80)
        print("Random raw split virtual-PNG pixel validation")
        print("=" * 80)
        print(f"Processor file: {Path(__file__).resolve()}")
        print("Render path: matplotlib Agg -> BytesIO virtual PNG -> RGB decode")
        print("Validated rendering stack: matplotlib==3.10.8, numpy==2.2.4, pillow==11.2.1")
        print(f"Raw split: {raw_split_dir.expanduser().resolve()}")
        print(f"YOLO split: {yolo_split_dir.expanduser().resolve()}")
        print(f"Samples:   {sample_count}/{len(samples)}")
        print(f"Seed:      {validation_seed}")
        print(f"Tolerance: max per-channel pixel diff <= {validation_pixel_tolerance}")
        print("=" * 80)

        for index, (split, stem, meta_path, data_path, yolo_image_path) in enumerate(selected, start=1):
            try:
                generated_png = preprocessor.raw_to_png_bytes(meta_path, data_path)
                if not generated_png.startswith(png_signature):
                    failures.append(f"{split}/{stem}: generated bytes do not start with PNG signature")
                    continue
                generated = preprocessor._png_bytes_to_rgb_array(generated_png)
                reference = read_reference_rgb(yolo_image_path)

                if generated.shape != (640, 640, 3):
                    failures.append(f"{split}/{stem}: bad generated shape {generated.shape}")
                if generated.dtype != np.uint8:
                    failures.append(f"{split}/{stem}: bad generated dtype {generated.dtype}")
                if reference.shape != generated.shape:
                    failures.append(
                        f"{split}/{stem}: shape mismatch generated={generated.shape}, "
                        f"reference={reference.shape}"
                    )
                    continue

                diff = np.abs(generated.astype(np.int16) - reference.astype(np.int16))
                max_diff = int(diff.max())
                mean_diff = float(diff.mean())
                if max_diff > validation_pixel_tolerance:
                    failures.append(
                        f"{split}/{stem}: generated virtual PNG differs from yolo_split reference; "
                        f"max_diff={max_diff}, mean_diff={mean_diff:.4f}, "
                        f"reference={yolo_image_path}"
                    )
            except Exception as exc:
                failures.append(f"{split}/{stem}: {exc}")

            if index % 10 == 0 or index == sample_count:
                print(f"Validated {index}/{sample_count} samples")

        if failures:
            print(format_errors("Virtual-PNG pixel validation failed", failures))
            return False

        print("Virtual-PNG pixel validation passed.")
        return True

    if run_mode == "none":
        print("RUN_MODE is 'none'; no action taken.")
        return

    if run_mode == "validate_random_raw_split":
        if not validate_random_raw_split_samples():
            sys.exit(1)
        return

    raise ValueError(f"Unsupported run_mode: {run_mode}")


if __name__ == "__main__":
    main()
