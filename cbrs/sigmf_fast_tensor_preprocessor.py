"""
Fast SigMF preprocessing that keeps matplotlib rendering but skips PNG encoding.

Pipeline:
    raw IQ -> spectrogram magnitude -> fftshift/log dB -> percentile clip
    -> matplotlib Agg canvas RGB -> BCHW float tensor

This keeps the output type expected by the existing YOLO/RT-DETR lower-level
model forward path: torch.float32, [1, 3, 640, 640], range [0, 1].

Direct test:
    python cbrs/sigmf_fast_tensor_preprocessor.py



Version warning:
    The trained model, yolo_split PNG generation, SigMF raw-file conversion, and
    inference code must use the same rendering stack. Pixel output is
    sensitive to matplotlib/numpy/Pillow versions. The currently validated
    versions are:

        matplotlib==3.10.8
        numpy==2.2.4
        pillow==11.2.1

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - exercised only when matplotlib is absent
    Figure = None
    FigureCanvasAgg = None

RAW_SIGMF_ROOT = PROJECT_ROOT / "raw_sigmf_split"
YOLO_SPLIT_ROOT = PROJECT_ROOT / "yolo_split"
ACTUAL_SAMPLE_RATE = 30.72e6
DEFAULT_NFFT = 512
DEFAULT_NOVERLAP = 384
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_PERCENTILE_CLIP = (5.0, 95.0)
DEFAULT_DPI = 100
DEFAULT_CMAP = "viridis"
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_MEAN_ABS_TOLERANCE = 0.1
DEFAULT_IMAGE_MAX_ABS_TOLERANCE = 1


def yolo_split_name(raw_split: str) -> str:
    return "val" if raw_split == "validation" else raw_split


def collect_sigmf_pairs(raw_root: Path, split: str, max_samples: int | None) -> list[tuple[Path, Path]]:
    split_dir = raw_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Raw split directory not found: {split_dir}")

    pairs: list[tuple[Path, Path]] = []
    for meta_path in sorted(split_dir.glob("*.sigmf-meta")):
        data_path = split_dir / meta_path.name.replace(".sigmf-meta", ".sigmf-data")
        if data_path.exists():
            pairs.append((meta_path, data_path))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    return pairs


def tensor_from_rgb_array(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float().div(255.0).unsqueeze(0)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def image_diff_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "mean_abs": None,
            "max_abs": None,
        }
    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": int(diff.max()),
    }


def expected_yolo_image_path(yolo_root: Path, raw_split: str, sample_id: str) -> Path:
    return yolo_root / yolo_split_name(raw_split) / "images" / f"{sample_id}.png"


class FastSigMFTensorPreprocessor:
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
        if Figure is None or FigureCanvasAgg is None:
            raise RuntimeError("matplotlib Agg is required for canvas preprocessing")
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.noverlap = noverlap
        self.image_size = image_size
        self.percentile_clip = percentile_clip
        self.cmap = cmap
        width_inch = image_size[0] / dpi
        height_inch = image_size[1] / dpi
        self.figure = Figure(figsize=(width_inch, height_inch), dpi=dpi, facecolor="black")
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.axes.axis("off")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.image_artist = None

    def raw_to_tensor(self, meta_path: str | Path, data_path: str | Path) -> torch.Tensor:
        return tensor_from_rgb_array(self.raw_to_rgb_array(meta_path, data_path))

    def raw_to_rgb_array(self, meta_path: str | Path, data_path: str | Path) -> np.ndarray:
        _ = meta_path
        iq_data = np.fromfile(Path(data_path), dtype=np.complex64)
        return self.iq_to_rgb_array(iq_data)

    def iq_to_tensor(self, iq_data: np.ndarray) -> torch.Tensor:
        return tensor_from_rgb_array(self.iq_to_rgb_array(iq_data))

    def iq_to_rgb_array(self, iq_data: np.ndarray) -> np.ndarray:
        sxx_db = self._iq_to_spectrogram_db(iq_data)
        return self._spectrogram_db_to_canvas_rgb(sxx_db)

    def _iq_to_spectrogram_db(self, iq_data: np.ndarray) -> np.ndarray:
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

    def _spectrogram_db_to_canvas_rgb(self, sxx_db: np.ndarray) -> np.ndarray:
        low, high = self.percentile_clip
        vmin = np.percentile(sxx_db, low)
        vmax = np.percentile(sxx_db, high)
        image_data = sxx_db.T
        if self.image_artist is None:
            self.image_artist = self.axes.imshow(
                image_data,
                aspect="auto",
                origin="upper",
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            self.image_artist.set_data(image_data)
            self.image_artist.set_clim(vmin, vmax)
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3].copy())


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_tensor(
    model_name: str,
    model,
    tensor: torch.Tensor,
    labels: torch.Tensor,
    metrics: Any,
    device: torch.device,
    conf_threshold: float,
    iou_threshold: float,
    postprocess_raw_output,
) -> tuple[dict[str, Any], float, float]:
    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw_output = model.model(tensor.to(device))
    synchronize_if_cuda(device)
    inference_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    detections, _backend = postprocess_raw_output(model_name, raw_output, conf_threshold, iou_threshold)
    sample_metrics = metrics.add(detections, labels)
    postprocess_sec = time.perf_counter() - t0
    return sample_metrics, inference_sec, postprocess_sec


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def run_recall_test(args: argparse.Namespace) -> dict[str, Any]:
    from local_sigmf_baseline import (
        DetectionMetricAccumulator,
        label_path_for_sample,
        load_detection_model,
        load_yolo_label_tensor,
        model_path_for_name,
        postprocess_raw_output,
        resolve_device,
        resolve_model_name,
    )

    model_name = resolve_model_name(args.model_name)
    model_path = args.model_path or model_path_for_name(model_name)
    device = resolve_device(args.device)
    model = load_detection_model(model_name, model_path, device)
    pairs = collect_sigmf_pairs(args.raw_root, args.split, args.samples)
    fast = FastSigMFTensorPreprocessor()
    fast_metrics = DetectionMetricAccumulator(f"{model_name}_fast_tensor")
    rows = []
    image_check_failures = []

    print(
        f"[fast-preprocess][recall] model={model_name} split={args.split} "
        f"samples={len(pairs)} device={device}"
    )
    for meta_path, data_path in pairs:
        sample_id = meta_path.name.replace(".sigmf-meta", "")
        labels = load_yolo_label_tensor(label_path_for_sample(args.yolo_root, args.split, sample_id))

        t0 = time.perf_counter()
        fast_rgb = fast.raw_to_rgb_array(meta_path, data_path)
        fast_tensor = tensor_from_rgb_array(fast_rgb)
        fast_preprocess_sec = time.perf_counter() - t0
        gt_image_path = expected_yolo_image_path(args.yolo_root, args.split, sample_id)
        image_diff = {
            "shape_match": False,
            "mean_abs": None,
            "max_abs": None,
            "pass": False,
            "error": "",
        }
        if gt_image_path.exists():
            image_diff.update(image_diff_stats(load_rgb_image(gt_image_path), fast_rgb))
            image_diff["pass"] = (
                bool(image_diff["shape_match"])
                and image_diff["mean_abs"] <= args.image_mean_abs_tolerance
                and image_diff["max_abs"] <= args.image_max_abs_tolerance
            )
        else:
            image_diff["error"] = f"missing image: {gt_image_path}"
        if not image_diff["pass"]:
            image_check_failures.append(sample_id)

        fast_sample_metrics, fast_inference_sec, fast_postprocess_sec = evaluate_tensor(
            model_name,
            model,
            fast_tensor,
            labels,
            fast_metrics,
            device,
            args.conf_threshold,
            args.iou_threshold,
            postprocess_raw_output,
        )

        row = {
            "sample_id": sample_id,
            "gt_count": int(labels.shape[0]),
            "fast_preprocess_sec": fast_preprocess_sec,
            "fast_inference_sec": fast_inference_sec,
            "fast_postprocess_sec": fast_postprocess_sec,
            "fast_recall50_sample": fast_sample_metrics["recall50"],
            "image_diff_mean_abs": image_diff["mean_abs"],
            "image_diff_max_abs": image_diff["max_abs"],
            "image_diff_pass": int(bool(image_diff["pass"])),
            "image_diff_error": image_diff["error"],
        }
        rows.append(row)
        print(
            f"[fast-preprocess][recall] {sample_id} "
            f"fast_pre={fast_preprocess_sec:.4f}s fast_R50={row['fast_recall50_sample']:.4f} "
            f"img_mean_abs={row['image_diff_mean_abs']} img_pass={row['image_diff_pass']}"
        )

    fast_summary = fast_metrics.summary()
    summary = {
        "mode": "recall_test",
        "model_name": model_name,
        "split": args.split,
        "samples": len(rows),
        "device": str(device),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "image_check": {
            "mean_abs_tolerance": args.image_mean_abs_tolerance,
            "max_abs_tolerance": args.image_max_abs_tolerance,
            "pass": len(image_check_failures) == 0,
            "failure_count": len(image_check_failures),
            "mean_abs": mean_numeric(rows, "image_diff_mean_abs"),
            "max_abs": max(
                [int(row["image_diff_max_abs"]) for row in rows if row.get("image_diff_max_abs") is not None],
                default=0,
            ),
            "failures": image_check_failures[:50],
        },
        "fast": {
            "recall": fast_summary["recall"],
            "precision": fast_summary["precision"],
            "map50": fast_summary["map50"],
            "map50_95": fast_summary["map50_95"],
            "mean_preprocess_sec": mean_numeric(rows, "fast_preprocess_sec"),
            "mean_inference_sec": mean_numeric(rows, "fast_inference_sec"),
            "mean_postprocess_sec": mean_numeric(rows, "fast_postprocess_sec"),
        },
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[fast-preprocess][recall] recall json written to {args.output_json}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast SigMF tensor preprocessing recall test")
    parser.add_argument("--model_name", default="cbrs_yolo8x")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--raw_root", type=Path, default=RAW_SIGMF_ROOT)
    parser.add_argument("--yolo_root", type=Path, default=YOLO_SPLIT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou_threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--image_mean_abs_tolerance", type=float, default=DEFAULT_IMAGE_MEAN_ABS_TOLERANCE)
    parser.add_argument("--image_max_abs_tolerance", type=int, default=DEFAULT_IMAGE_MAX_ABS_TOLERANCE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", type=Path, default=PROJECT_ROOT / "FastSigmf_recall_test.json")
    return parser.parse_args()


def main() -> None:
    summary = run_recall_test(parse_args())
    print(
        "[fast-preprocess][recall] summary "
        f"samples={summary['samples']} "
        f"fast_recall={summary['fast']['recall']:.4f} "
        f"fast_pre={summary['fast']['mean_preprocess_sec']:.4f}s "
        f"image_pass={summary['image_check']['pass']}"
    )


if __name__ == "__main__":
    main()
