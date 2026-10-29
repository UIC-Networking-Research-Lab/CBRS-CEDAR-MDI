"""
CBRS RT-DETR training script.

Dataset splitting is handled by split_yolo_dataset.py.
This script reads an existing yolo_split/data.yaml and trains Ultralytics RT-DETR
models serially. Edit the globals below, then run:

    python scripts/yolo/train_rtdetr.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

import gc
import sys
# VERSION WARNING:
# The trained model depends on the exact yolo_split PNG rendering distribution.
# Dataset PNG generation, training, SigMF raw conversion, and inference
# must use the same rendering stack:
#   matplotlib==3.10.8
#   numpy==2.2.4
#   pillow==11.2.1
# This version set has passed virtual-PNG pixel validation against yolo_split.

from pathlib import Path

from tqdm import tqdm
from scripts.data.split_yolo_dataset import validate_split_yaml


# Edit these globals directly before running this script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "yolo_split"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models_retrained"

RUN_MODE = "sweep"  # "train" or "sweep"
EPOCHS = 25
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = None  # None=Ultralytics auto, "0"=GPU 0, "cpu"=CPU
CACHE = "disk"
WORKERS = 12

# Single-model training config, used when RUN_MODE = "train".
MODEL_NAME = "rtdetr-l.pt"
RUN_NAME = "cbrs_rtdetr_l"

# Serial sweep config, used when RUN_MODE = "sweep".
SWEEP_MODELS = ["rtdetr-l.pt", "rtdetr-x.pt"]
SKIP_MISSING_LOCAL = False


def resolve_model_path(model_name):
    """Prefer a local weight file next to this script when it exists."""
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    local_model_path = PROJECT_ROOT / model_name
    if local_model_path.exists():
        return str(local_model_path)

    return model_name


def local_weight_exists(model_name):
    return Path(model_name).exists() or (PROJECT_ROOT / model_name).exists()


def run_name_for(model_name):
    stem = Path(model_name).stem.lower().replace("-", "_")
    if stem.startswith("rtdetr_"):
        stem = stem[len("rtdetr_") :]
    return f"cbrs_rtdetr_{stem}"


def attach_progress_callbacks(model, run_name, total_epochs):
    """Show one stable epoch progress bar for a serial RT-DETR run."""
    progress_bar = tqdm(
        total=total_epochs,
        desc=run_name,
        unit="epoch",
        dynamic_ncols=True,
    )
    state = {"last_epoch": 0}

    def on_train_start(trainer):
        trainer_epochs = getattr(trainer, "epochs", None)
        if trainer_epochs and int(trainer_epochs) != progress_bar.total:
            progress_bar.total = int(trainer_epochs)
            progress_bar.refresh()

    def on_train_epoch_end(trainer):
        epoch = getattr(trainer, "epoch", None)
        if epoch is None:
            progress_bar.update(1)
            return

        current = int(epoch) + 1
        delta = current - state["last_epoch"]
        if delta > 0:
            progress_bar.update(delta)
            state["last_epoch"] = current

    def on_train_end(_trainer):
        remaining = (progress_bar.total or 0) - progress_bar.n
        if remaining > 0:
            progress_bar.update(remaining)

    try:
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)
    except AttributeError:
        tqdm.write("Current ultralytics version does not expose training callbacks.")

    return progress_bar


def cleanup_after_train():
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    yaml_path=None,
    model_name=None,
    run_name=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    quiet=False,
):
    """Train one Ultralytics RT-DETR model."""
    try:
        from ultralytics import RTDETR
    except ImportError:
        print("Error: ultralytics is not installed.")
        print("Install it with: pip install ultralytics")
        sys.exit(1)

    yaml_path = yaml_path or validate_split_yaml(output_dir=OUTPUT_DIR)
    model_name = model_name or MODEL_NAME
    run_name = run_name or RUN_NAME
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE
    img_size = img_size or IMG_SIZE
    device = DEVICE if device is None else device

    if not yaml_path.exists():
        print(f"Error: data.yaml not found: {yaml_path}")
        print("Please run first: python scripts/data/split_yolo_dataset.py")
        sys.exit(1)

    if not quiet:
        print("=" * 60)
        print("RT-DETR training")
        print("=" * 60)
        print("Config:")
        print(f"  Model: {model_name}")
        print(f"  Run Name: {run_name}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Image Size: {img_size}")
        print(f"  Device: {device if device is not None else 'auto'}")
        print(f"  Cache: {CACHE}")
        print(f"  Workers: {WORKERS}")
        print(f"  Data: {yaml_path}")

    model = RTDETR(resolve_model_path(model_name))
    progress_bar = attach_progress_callbacks(model, run_name, epochs)

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(TRAINED_MODELS_DIR),
        name=run_name,
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        cache=CACHE,
        workers=WORKERS,
        verbose=False,
    )
    if device is not None:
        train_kwargs["device"] = device

    try:
        results = model.train(**train_kwargs)
    finally:
        progress_bar.close()
        cleanup_after_train()

    if not quiet:
        best_path = TRAINED_MODELS_DIR / run_name / "weights" / "best.pt"
        print("\nTraining complete.")
        print(f"Best model: {best_path}")

    return results


def sweep_models(
    yaml_path=None,
    models=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    skip_missing_local=False,
):
    if models is None:
        models = SWEEP_MODELS

    if yaml_path is None:
        yaml_path = validate_split_yaml(output_dir=OUTPUT_DIR)

    print("=" * 70)
    print("RT-DETR sweep plan")
    print("=" * 70)
    print(f"Data:   {yaml_path}")
    print(f"Output: {TRAINED_MODELS_DIR}")
    print(f"Models: {', '.join(models)}")
    print("Mode:   serial, no parallel training")
    print("=" * 70)

    jobs = []
    skipped = []
    for index, model_name in enumerate(models, start=1):
        if skip_missing_local and not local_weight_exists(model_name):
            skipped.append(model_name)
            print(f"\n[{index}/{len(models)}] Skip missing local weight: {model_name}")
            continue

        jobs.append((index, model_name, run_name_for(model_name)))

    if not jobs:
        print("No sweep jobs to run.")
        return []

    failures = []
    for completed, (index, model_name, run_name) in enumerate(jobs, start=1):
        print("\n" + "-" * 70)
        print(f"[{index}/{len(models)}] Start {model_name} -> {run_name}")
        print("-" * 70)

        try:
            train_model(
                yaml_path=yaml_path,
                model_name=model_name,
                run_name=run_name,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                device=device,
                quiet=True,
            )
            print(f"Completed [{completed}/{len(jobs)}]: {model_name} -> {run_name}")
        except Exception as exc:
            failures.append((model_name, str(exc)))
            print(f"Failed: {model_name}: {exc}")
        finally:
            cleanup_after_train()

    print("\n" + "=" * 70)
    print("RT-DETR sweep complete")
    print("=" * 70)
    if failures:
        print(f"Failures: {len(failures)}")
        for model_name, error in failures:
            print(f"  {model_name}: {error}")
    else:
        print("All requested runs completed.")
    if skipped:
        print(f"Skipped missing local weights: {len(skipped)}")

    return failures


def main():
    yaml_path = validate_split_yaml(output_dir=OUTPUT_DIR)

    if RUN_MODE == "train":
        train_model(
            yaml_path=yaml_path,
            model_name=MODEL_NAME,
            run_name=RUN_NAME,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=DEVICE,
        )
        return

    if RUN_MODE == "sweep":
        sweep_models(
            yaml_path=yaml_path,
            models=SWEEP_MODELS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=DEVICE,
            skip_missing_local=SKIP_MISSING_LOCAL,
        )
        return

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
