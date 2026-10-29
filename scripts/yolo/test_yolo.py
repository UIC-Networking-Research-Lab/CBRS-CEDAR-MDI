"""
CBRS dataset - test script for the Ultralytics detectors (YOLO / RT-DETR).

1. Evaluates detection quality on the test split (mAP, precision, recall).
2. Detects LTE and radar boxes.
3. Compares the frame-level overlap decision with the ground truth.
4. Saves annotated prediction images.

The input is the prepared PNG split (yolo_split/). For the end-to-end pipeline
from raw SigMF I/Q files, use local_sigmf_baseline.py.

Usage (from the repository root):
    python scripts/yolo/test_yolo.py                                   # every model under trained_models/
    python scripts/yolo/test_yolo.py --models trained_models/cbrs_yolo8n --no-predict
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import argparse
import json
import csv
# VERSION WARNING:
# Ultralytics evaluation/inference consumes PNG/image inputs. Dataset PNG
# generation, trained models, SigMF raw conversion, and inference must
# use the same rendering stack:
#   matplotlib==3.10.8
#   numpy==2.2.4
#   pillow==11.2.1
# This version set has passed virtual-PNG pixel validation against yolo_split.

from pathlib import Path

import numpy as np
from tqdm import tqdm

# ============== Configuration ==============

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "trained_models"
OUTPUT_ROOT = PROJECT_ROOT / "yolo_results"
MODEL_WEIGHT_NAMES = ("best.pt", "last.pt")
IMAGE_PATH = None
RUN_PREDICTION = True
SAVE_RESULTS = True


def find_dataset_dir():
    candidates = [
        PROJECT_ROOT / "yolo_split",
        Path.cwd() / "yolo_split",
    ]

    for candidate in candidates:
        if (candidate / "data.yaml").exists():
            return candidate.resolve()

    for candidate in PROJECT_ROOT.iterdir():
        if candidate.is_dir() and (candidate / "data.yaml").exists():
            return candidate.resolve()

    return (PROJECT_ROOT / "yolo_split").resolve()


def resolve_model_path(model_path):
    model_path = Path(model_path).expanduser()
    if model_path.is_dir():
        weight_path = find_model_weight(model_path)
        if weight_path is not None:
            return weight_path
        return model_path / "weights" / MODEL_WEIGHT_NAMES[0]
    return model_path


def find_model_weight(model_dir):
    weights_dir = Path(model_dir).expanduser() / "weights"
    for weight_name in MODEL_WEIGHT_NAMES:
        weight_path = weights_dir / weight_name
        if weight_path.exists():
            return weight_path
    return None


def discover_model_paths(model_root):
    model_root = Path(model_root).expanduser()

    if model_root.is_file():
        return [model_root.resolve()]

    if not model_root.exists():
        return []

    direct_weight = find_model_weight(model_root)
    if direct_weight is not None:
        return [direct_weight.resolve()]

    model_files = []
    seen = set()

    def add_model(model_file):
        resolved = model_file.resolve()
        if resolved not in seen:
            seen.add(resolved)
            model_files.append(resolved)

    for pt_file in sorted(model_root.glob("*.pt")):
        add_model(pt_file)

    for child_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        if child_dir.name.startswith("sig_yolo_"):
            continue  # CEDAR-YOLO models: use scripts/cedar/test_cedar_yolo.py
        weight_path = find_model_weight(child_dir)
        if weight_path is not None:
            add_model(weight_path)

    return model_files


def model_output_name(model_path):
    model_path = Path(model_path).expanduser()
    if model_path.is_dir():
        return model_path.name
    if model_path.name in ("best.pt", "last.pt") and model_path.parent.name == "weights":
        return model_path.parent.parent.name
    return model_path.stem


def load_ultralytics_model(model_file):
    model_text = str(model_file).lower()
    try:
        if "rtdetr" in model_text:
            from ultralytics import RTDETR

            return RTDETR(str(model_file))

        from ultralytics import YOLO

        return YOLO(str(model_file))
    except ImportError:
        print("Error: the ultralytics package is not installed")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


DATASET_DIR = find_dataset_dir()

# Inference parameters
CONF_THRESHOLD = 0.25  # confidence threshold
IOU_THRESHOLD = 0.45   # NMS IoU threshold
IMG_SIZE = 640

# Classes
CLASSES = {
    0: "LTE",
    1: "Radar"
}


def check_frequency_overlap(lte_box, radar_box):
    """
    Check whether LTE and radar overlap along the frequency (x) axis.

    box format: [x_center, y_center, width, height] (normalized to 0-1)

    Returns:
        is_overlap: bool
        overlap_ratio: float, overlap width relative to the narrower signal
    """
    # Frequency extents (x axis)
    lte_left = lte_box[0] - lte_box[2] / 2
    lte_right = lte_box[0] + lte_box[2] / 2

    radar_left = radar_box[0] - radar_box[2] / 2
    radar_right = radar_box[0] + radar_box[2] / 2

    # Intersection
    overlap_left = max(lte_left, radar_left)
    overlap_right = min(lte_right, radar_right)

    if overlap_left >= overlap_right:
        return False, 0.0

    overlap_width = overlap_right - overlap_left

    # Overlap ratio relative to the narrower signal
    min_width = min(lte_box[2], radar_box[2])
    overlap_ratio = overlap_width / min_width if min_width > 0 else 0

    return True, overlap_ratio


def analyze_detections(detections):
    """
    Analyze detections and decide the overlap state.

    Args:
        detections: Ultralytics result for one image

    Returns:
        analysis: dict
    """
    lte_boxes = []
    radar_boxes = []

    if detections is None or len(detections) == 0:
        return {
            'lte_count': 0,
            'radar_count': 0,
            'has_overlap': False,
            'overlap_pairs': [],
            'lte_boxes': [],
            'radar_boxes': []
        }

    # Boxes (format: [x_center, y_center, width, height])
    boxes = detections.boxes

    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        # Normalized xywh
        xywh = boxes.xywhn[i].cpu().numpy()
        box_data = {
            'box': xywh.tolist(),
            'confidence': conf
        }

        if cls == 0:  # LTE
            lte_boxes.append(box_data)
        elif cls == 1:  # Radar
            radar_boxes.append(box_data)

    # Overlap check
    overlap_pairs = []
    has_overlap = False

    for i, lte in enumerate(lte_boxes):
        for j, radar in enumerate(radar_boxes):
            is_overlap, ratio = check_frequency_overlap(lte['box'], radar['box'])
            if is_overlap:
                has_overlap = True
                overlap_pairs.append({
                    'lte_idx': i,
                    'radar_idx': j,
                    'overlap_ratio': ratio
                })

    return {
        'lte_count': len(lte_boxes),
        'radar_count': len(radar_boxes),
        'has_overlap': has_overlap,
        'overlap_pairs': overlap_pairs,
        'lte_boxes': lte_boxes,
        'radar_boxes': radar_boxes
    }


def load_ground_truth_labels(label_path):
    """
    Load the ground truth from a YOLO label file.

    Args:
        label_path: label file path (.txt)

    Returns:
        dict with lte_boxes, radar_boxes and has_overlap
    """
    lte_boxes = []
    radar_boxes = []

    if not label_path.exists():
        return {
            'lte_count': 0,
            'radar_count': 0,
            'has_overlap': False,
            'lte_boxes': [],
            'radar_boxes': []
        }

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                # YOLO format: class x_center y_center width height
                box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]

                if cls == 0:  # LTE
                    lte_boxes.append({'box': box})
                elif cls == 1:  # Radar
                    radar_boxes.append({'box': box})

    # Ground-truth overlap state
    has_overlap = False
    for lte in lte_boxes:
        for radar in radar_boxes:
            is_overlap, _ = check_frequency_overlap(lte['box'], radar['box'])
            if is_overlap:
                has_overlap = True
                break
        if has_overlap:
            break

    return {
        'lte_count': len(lte_boxes),
        'radar_count': len(radar_boxes),
        'has_overlap': has_overlap,
        'lte_boxes': lte_boxes,
        'radar_boxes': radar_boxes
    }


def evaluate_model(model_path=None, output_dir=None, device=None):
    """
    Evaluate a model on the test split.
    """
    model_file = resolve_model_path(model_path or MODEL_PATH)
    run_output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / model_output_name(model_file)
    if not model_file.exists():
        print(f"Error: model not found: {model_file}")
        print("Download the released weights or train first: python scripts/yolo/train_yolo.py")
        sys.exit(1)

    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"Error: dataset config not found: {yaml_path}")
        sys.exit(1)

    print("="*60)
    print("Model evaluation - test split")
    print("="*60)
    print(f"Model: {model_file}")
    print(f"Output: {run_output_dir}")

    # Load model
    model = load_ultralytics_model(model_file)

    # Evaluate
    metrics = model.val(
        device=device,
        data=str(yaml_path),
        split='test',
        imgsz=IMG_SIZE,
        batch=16,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        project=str(run_output_dir),
        name="val",
        exist_ok=True,
        verbose=True,
        plots=True,
        save_json=True
    )

    print("\n" + "="*60)
    print("Evaluation results:")
    print("="*60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")

    print("\nPer-class AP@50:")
    for i, name in CLASSES.items():
        if i < len(metrics.box.ap50):
            print(f"  {name}: {metrics.box.ap50[i]:.4f}")

    eval_summary = {
        "model": str(model_file),
        "dataset": str(DATASET_DIR),
        "split": "test",
        "detection": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "ap50_by_class": {
                name: float(metrics.box.ap50[i])
                for i, name in CLASSES.items()
                if i < len(metrics.box.ap50)
            },
        },
        "overlap": None,
    }

    # ===== Overlap analysis (compared with the ground truth) =====
    print("\n" + "="*60)
    print("Overlap decision - compared with the ground truth")
    print("="*60)

    # Overlap analysis on the test split
    test_img_dir = DATASET_DIR / "test" / "images"
    test_label_dir = DATASET_DIR / "test" / "labels"

    if test_img_dir.exists() and test_label_dir.exists():
        images = sorted(test_img_dir.glob("*.png"))

        # Ground-truth counts
        gt_stats = {'overlap': 0, 'no_overlap': 0}
        # Prediction counts
        pred_stats = {'overlap': 0, 'no_overlap': 0}

        # Overlap confusion matrix (overlap = positive class)
        # TP: GT overlap,    predicted overlap
        # TN: GT no overlap, predicted no overlap
        # FP: GT no overlap, predicted overlap (false alarm)
        # FN: GT overlap,    predicted no overlap (miss)
        confusion = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

        overlap_ratios = []

        print(f"Analyzing {len(images)} test images...")

        for img_path in tqdm(images, desc="Overlap eval", unit="img"):
            base_name = img_path.stem
            label_path = test_label_dir / f"{base_name}.txt"

            # 1. Ground truth
            gt = load_ground_truth_labels(label_path)
            gt_has_overlap = gt['has_overlap']

            if gt_has_overlap:
                gt_stats['overlap'] += 1
            else:
                gt_stats['no_overlap'] += 1

            # 2. Prediction
            results = model.predict(
                device=device,
                source=str(img_path),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )

            pred = analyze_detections(results[0])
            pred_has_overlap = pred['has_overlap']

            if pred_has_overlap:
                pred_stats['overlap'] += 1
            else:
                pred_stats['no_overlap'] += 1

            # 3. Confusion matrix
            if gt_has_overlap and pred_has_overlap:
                confusion['TP'] += 1
            elif not gt_has_overlap and not pred_has_overlap:
                confusion['TN'] += 1
            elif not gt_has_overlap and pred_has_overlap:
                confusion['FP'] += 1
            else:  # gt_has_overlap and not pred_has_overlap
                confusion['FN'] += 1

            # Overlap ratios
            for pair in pred['overlap_pairs']:
                overlap_ratios.append(pair['overlap_ratio'])

        total = len(images)

        # Ground-truth distribution
        print("\n[Ground-truth distribution]")
        print(f"  Overlap:     {gt_stats['overlap']} ({gt_stats['overlap']/total*100:.1f}%)")
        print(f"  No overlap:  {gt_stats['no_overlap']} ({gt_stats['no_overlap']/total*100:.1f}%)")

        # Prediction distribution
        print("\n[Prediction distribution]")
        print(f"  Overlap:     {pred_stats['overlap']} ({pred_stats['overlap']/total*100:.1f}%)")
        print(f"  No overlap:  {pred_stats['no_overlap']} ({pred_stats['no_overlap']/total*100:.1f}%)")

        # Confusion matrix
        print("\n[Overlap confusion matrix]")
        print("                    Pred overlap   Pred no overlap")
        print(f"  GT Overlap:         {confusion['TP']:5d}           {confusion['FN']:5d}")
        print(f"  GT NoOverlap:       {confusion['FP']:5d}           {confusion['TN']:5d}")

        # Metrics
        TP, TN, FP, FN = confusion['TP'], confusion['TN'], confusion['FP'], confusion['FN']

        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("\n[Overlap decision metrics]")
        print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:    {f1:.4f}")

        if overlap_ratios:
            print("\n[Predicted overlap ratio]")
            print(f"  Mean:  {np.mean(overlap_ratios):.2%}")
            print(f"  Max:   {np.max(overlap_ratios):.2%}")
            print(f"  Min:   {np.min(overlap_ratios):.2%}")

        eval_summary["overlap"] = {
                "ground_truth": gt_stats,
                "prediction": pred_stats,
                "confusion": confusion,
                "TP": int(TP),
                "TN": int(TN),
                "FP": int(FP),
                "FN": int(FN),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "overlap_accuracy": float(accuracy),
                "overlap_precision": float(precision),
                "overlap_recall": float(recall),
                "overlap_f1": float(f1),
                "overlap_ratio": {
                    "mean": float(np.mean(overlap_ratios)) if overlap_ratios else None,
                    "max": float(np.max(overlap_ratios)) if overlap_ratios else None,
                    "min": float(np.min(overlap_ratios)) if overlap_ratios else None,
                },
            }

        output_json_dir = run_output_dir / "analysis"
        output_json_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_json_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation summary saved to: {summary_path}")

    return eval_summary


def predict_with_overlap(image_path=None, save_results=True, model_path=None, output_dir=None, device=None):
    """
    Run prediction, decide the overlap state and save annotated images.
    """
    try:
        import cv2
    except ImportError:
        print("Error: missing dependencies")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    model_file = resolve_model_path(model_path or MODEL_PATH)
    run_output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / model_output_name(model_file)
    if not model_file.exists():
        print(f"Error: model not found: {model_file}")
        sys.exit(1)

    # Images to process
    if image_path:
        images = [Path(image_path)]
    else:
        # Default: test split
        test_dir = DATASET_DIR / "test" / "images"
        if not test_dir.exists():
            print(f"Error: test split not found: {test_dir}")
            sys.exit(1)
        images = sorted(test_dir.glob("*.png"))

    print("="*60)
    print("Prediction and overlap decision")
    print("="*60)
    print(f"Model: {model_file}")
    print(f"Output: {run_output_dir}")
    print(f"Images: {len(images)}")

    # Output directories
    output_images_dir = run_output_dir / "predictions"
    output_json_dir = run_output_dir / "analysis"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_ultralytics_model(model_file)

    # Counters
    total = 0
    overlap_count = 0
    results_summary = []

    for img_path in tqdm(images, desc="Predict + visualize", unit="img"):
        total += 1

        # Prediction
        results = model.predict(
            device=device,
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        # Overlap analysis
        analysis = analyze_detections(results[0])
        analysis['image'] = img_path.name

        if analysis['has_overlap']:
            overlap_count += 1

        results_summary.append(analysis)

        # Save the annotated image
        if save_results:
            # Read image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # LTE boxes (blue)
            for lte in analysis['lte_boxes']:
                box = lte['box']
                x1 = int((box[0] - box[2]/2) * w)
                y1 = int((box[1] - box[3]/2) * h)
                x2 = int((box[0] + box[2]/2) * w)
                y2 = int((box[1] + box[3]/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(img, f"LTE {lte['confidence']:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

            # Radar boxes (red)
            for radar in analysis['radar_boxes']:
                box = radar['box']
                x1 = int((box[0] - box[2]/2) * w)
                y1 = int((box[1] - box[3]/2) * h)
                x2 = int((box[0] + box[2]/2) * w)
                y2 = int((box[1] + box[3]/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Radar {radar['confidence']:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Overlap state
            status = "OVERLAP" if analysis['has_overlap'] else "NO OVERLAP"
            color = (0, 0, 255) if analysis['has_overlap'] else (0, 255, 0)
            cv2.putText(img, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Save
            output_path = output_images_dir / img_path.name
            cv2.imwrite(str(output_path), img)

    # Save the analysis
    summary_path = output_json_dir / "overlap_analysis.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\nSummary:")
    print(f"  Images:     {total}")
    print(f"  Overlap:    {overlap_count} ({overlap_count/total*100:.1f}%)")
    print(f"  No overlap: {total - overlap_count} ({(total-overlap_count)/total*100:.1f}%)")
    print(f"\nResults saved to: {run_output_dir}")

    return results_summary


SUMMARY_COLUMNS = [
    ("model", "Model"),
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("det_precision", "DetP"),
    ("det_recall", "DetR"),
    ("gt_overlap", "GT_O"),
    ("gt_no_overlap", "GT_N"),
    ("gt_overlap_rate", "GT_O%"),
    ("pred_overlap", "Pred_O"),
    ("pred_no_overlap", "Pred_N"),
    ("pred_overlap_rate", "Pred_O%"),
    ("tp", "TP"),
    ("fn", "FN"),
    ("fp", "FP"),
    ("tn", "TN"),
    ("overlap_accuracy", "Acc"),
    ("overlap_precision", "Prec"),
    ("overlap_recall", "Rec"),
    ("overlap_f1", "F1"),
    ("overlap_mean", "MeanOv"),
    ("overlap_max", "MaxOv"),
    ("overlap_min", "MinOv"),
]
SUMMARY_PERCENT_FIELDS = {
    "gt_overlap_rate",
    "pred_overlap_rate",
    "overlap_accuracy",
    "overlap_precision",
    "overlap_recall",
    "overlap_f1",
    "overlap_mean",
    "overlap_max",
    "overlap_min",
}


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def flatten_eval_summary(summary):
    detection = summary.get("detection") or {}
    overlap = summary.get("overlap") or {}
    gt_stats = overlap.get("ground_truth") or {}
    pred_stats = overlap.get("prediction") or {}
    confusion = overlap.get("confusion") or {}
    overlap_ratio = overlap.get("overlap_ratio") or {}

    gt_overlap = gt_stats.get("overlap")
    gt_no_overlap = gt_stats.get("no_overlap")
    pred_overlap = pred_stats.get("overlap")
    pred_no_overlap = pred_stats.get("no_overlap")
    gt_total = (gt_overlap or 0) + (gt_no_overlap or 0)
    pred_total = (pred_overlap or 0) + (pred_no_overlap or 0)

    model_path = Path(summary.get("model", ""))
    return {
        "model": model_output_name(model_path) if summary.get("model") else "",
        "model_path": summary.get("model", ""),
        "map50": detection.get("map50"),
        "map50_95": detection.get("map50_95"),
        "det_precision": detection.get("precision"),
        "det_recall": detection.get("recall"),
        "gt_overlap": gt_overlap,
        "gt_no_overlap": gt_no_overlap,
        "gt_overlap_rate": safe_rate(gt_overlap or 0, gt_total),
        "pred_overlap": pred_overlap,
        "pred_no_overlap": pred_no_overlap,
        "pred_overlap_rate": safe_rate(pred_overlap or 0, pred_total),
        "tp": confusion.get("TP"),
        "fn": confusion.get("FN"),
        "fp": confusion.get("FP"),
        "tn": confusion.get("TN"),
        "overlap_accuracy": overlap.get("overlap_accuracy", overlap.get("accuracy")),
        "overlap_precision": overlap.get("overlap_precision", overlap.get("precision")),
        "overlap_recall": overlap.get("overlap_recall", overlap.get("recall")),
        "overlap_f1": overlap.get("overlap_f1", overlap.get("f1")),
        "overlap_mean": overlap_ratio.get("mean"),
        "overlap_max": overlap_ratio.get("max"),
        "overlap_min": overlap_ratio.get("min"),
    }


def format_summary_value(key, value):
    if value is None:
        return "-"
    if key in SUMMARY_PERCENT_FIELDS:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_overall_summary_table(rows):
    if not rows:
        return

    formatted_rows = [
        [format_summary_value(key, row.get(key)) for key, _header in SUMMARY_COLUMNS]
        for row in rows
    ]
    headers = [header for _key, header in SUMMARY_COLUMNS]
    widths = [
        max(len(header), *(len(row[index]) for row in formatted_rows))
        for index, header in enumerate(headers)
    ]

    print("\n" + "=" * 120)
    print("All model summary")
    print("=" * 120)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def save_overall_summary(raw_summaries, rows, output_root):
    summary_dir = Path(output_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = summary_dir / "all_models_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_summaries, f, indent=2, ensure_ascii=False)

    table_json_path = summary_dir / "all_models_summary_table.json"
    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = summary_dir / "all_models_evaluation_summary.csv"
    csv_fields = [key for key, _header in SUMMARY_COLUMNS] + ["model_path"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    print(f"\nOverall summary saved to: {json_path}")
    print(f"Overall summary table JSON saved to: {table_json_path}")
    print(f"Overall summary CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ultralytics CBRS detectors on yolo_split PNGs")
    parser.add_argument("--models", type=Path, default=MODEL_PATH,
                        help="A .pt file, a run directory with weights/best.pt, or a parent directory of runs")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="Output root directory")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0; default: Ultralytics selection")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH,
                        help="Predict a single image instead of the whole test split")
    parser.add_argument("--no-predict", action="store_true",
                        help="Skip the annotated prediction images")
    args = parser.parse_args()

    model_paths = discover_model_paths(args.models)
    output_root = Path(args.output).expanduser()

    if not model_paths:
        print(f"Error: no model files found under {Path(args.models).expanduser()}")
        print("Expected a .pt file, a run directory with weights/best.pt, or a parent directory of runs.")
        sys.exit(1)

    print("\n" + "#"*60)
    print("# CBRS Ultralytics model test pipeline")
    print(f"# Models found: {len(model_paths)}")
    print(f"# Output root: {output_root}")
    print("#"*60)

    eval_summaries = []
    for index, model_file in enumerate(model_paths, start=1):
        run_name = model_output_name(model_file)
        run_output_dir = output_root / run_name

        print("\n" + "#"*60)
        print(f"# [{index}/{len(model_paths)}] {run_name}")
        print("#"*60)

        if args.image is not None:
            predict_with_overlap(image_path=args.image, save_results=SAVE_RESULTS,
                                 model_path=model_file, output_dir=run_output_dir, device=args.device)
            continue
        eval_summary = evaluate_model(model_path=model_file, output_dir=run_output_dir, device=args.device)
        eval_summaries.append(eval_summary)

        if RUN_PREDICTION and not args.no_predict:
            print("\n")
            predict_with_overlap(
                image_path=args.image,
                save_results=SAVE_RESULTS,
                model_path=model_file,
                output_dir=run_output_dir,
                device=args.device,
            )

    summary_rows = [flatten_eval_summary(summary) for summary in eval_summaries]
    print_overall_summary_table(summary_rows)
    save_overall_summary(eval_summaries, summary_rows, output_root)

    print("\n" + "#"*60)
    print("# All tests complete")
    print("#"*60)


if __name__ == "__main__":
    main()
