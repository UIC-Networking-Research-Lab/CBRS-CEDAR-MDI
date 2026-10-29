"""
CBRS YOLO training script.

Dataset splitting is handled by split_yolo_dataset.py.
This script reads an existing yolo_split/data.yaml for train or sweep.

Usage:
    python scripts/data/split_yolo_dataset.py
    python scripts/yolo/train_yolo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue

from tqdm import tqdm
from scripts.data.split_yolo_dataset import validate_split_yaml

# Edit these globals directly before running this script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "yolo_split"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models_retrained"

RUN_MODE = "sweep"  # "train" or "sweep"
# EPOCHS / BATCH_SIZE match the released checkpoints (trained_models/<run>/args.yaml).
EPOCHS = 25
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = None  # None=Ultralytics auto, "0"=GPU 0, "cpu"=CPU
CACHE = "disk"
WORKERS = 12

# Single-model training config, used when RUN_MODE = "train".
MODEL_NAME = "yolo11x.pt"
RUN_NAME = "cbrs_yolo11x"

# Sweep config, used when RUN_MODE = "sweep".
SWEEP_VERSIONS = ["v8", "v11", "v26"]
SWEEP_SCALES = ["n", "s", "x"]
SWEEP_PARALLEL = 1
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


def normalize_version(version):
    version = version.lower().strip()
    if version.startswith("yolo"):
        version = version[4:]
    if not version.startswith("v"):
        version = f"v{version}"
    return version


def model_name_for(version, scale):
    version = normalize_version(version)
    scale = scale.lower().strip()

    if version == "v8":
        return f"yolov8{scale}.pt"

    return f"yolo{version[1:]}{scale}.pt"


def run_name_for(version, scale):
    version = normalize_version(version)
    return f"cbrs_yolo{version[1:]}{scale}"


def local_weight_exists(model_name):
    return Path(model_name).exists() or (PROJECT_ROOT / model_name).exists()


def attach_progress_callbacks(model, job_id, total_epochs, progress_queue):
    """Send Ultralytics epoch progress back to the sweep coordinator."""
    if progress_queue is None:
        return

    def emit(event, **payload):
        progress_queue.put((event, job_id, payload))

    def on_train_start(trainer):
        trainer_epochs = getattr(trainer, "epochs", None) or total_epochs
        emit("start", total=int(trainer_epochs))

    def on_train_epoch_end(trainer):
        epoch = getattr(trainer, "epoch", None)
        current = int(epoch) + 1 if epoch is not None else None
        emit("epoch", current=current)

    def on_train_end(_trainer):
        emit("done")

    try:
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)
    except AttributeError:
        emit("message", text="current ultralytics version does not expose training callbacks")


def train_model(
    yaml_path=None,
    model_name=None,
    run_name=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    progress_queue=None,
    progress_job_id=None,
    quiet=False,
):
    """Train one YOLO model."""
    try:
        from ultralytics import YOLO
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
        print("YOLO training")
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

    model = YOLO(resolve_model_path(model_name))
    attach_progress_callbacks(model, progress_job_id or run_name, epochs, progress_queue)

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
        verbose=progress_queue is None,
    )
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    if not quiet:
        best_path = TRAINED_MODELS_DIR / run_name / "weights" / "best.pt"
        print("\nTraining complete.")
        print(f"Best model: {best_path}")

    return results


def sweep_models(
    yaml_path=None,
    versions=None,
    scales=None,
    epochs=None,
    batch_size=None,
    img_size=None,
    device=None,
    skip_missing_local=False,
    parallel=SWEEP_PARALLEL,
):
    if versions is None:
        versions = SWEEP_VERSIONS
    if scales is None:
        scales = SWEEP_SCALES

    if yaml_path is None:
        yaml_path = validate_split_yaml(output_dir=OUTPUT_DIR)

    versions = [normalize_version(v) for v in versions]
    scales = [s.lower().strip() for s in scales]
    planned = [(version, scale, model_name_for(version, scale)) for version in versions for scale in scales]

    print("=" * 70)
    print("YOLO sweep plan")
    print("=" * 70)
    print(f"Data:   {yaml_path}")
    print(f"Output: {TRAINED_MODELS_DIR}")
    print(f"Models: {', '.join(model for _, _, model in planned)}")
    print("=" * 70)

    jobs = []
    skipped = []
    for index, (version, scale, model_name) in enumerate(planned, start=1):
        if skip_missing_local and not local_weight_exists(model_name):
            skipped.append(model_name)
            print(f"\n[{index}/{len(planned)}] Skip missing local weight: {model_name}")
            continue

        jobs.append((index, version, scale, model_name, run_name_for(version, scale)))

    failures = []
    completed = 0
    worker_count = max(1, min(parallel, len(jobs))) if jobs else 0
    print(f"Parallel training workers: {worker_count} / requested {parallel}")

    if not jobs:
        print("No sweep jobs to run.")
        return failures

    progress_queue = Queue()

    def drain_progress(progress_bar, last_epochs, total_epochs_by_job, active_jobs):
        while True:
            try:
                event, job_id, payload = progress_queue.get_nowait()
            except Empty:
                break

            if event == "start":
                total = payload.get("total")
                active_jobs.add(job_id)
                if total:
                    total = int(total)
                    old_total = total_epochs_by_job.get(job_id, epochs or EPOCHS)
                    total_epochs_by_job[job_id] = total
                    if total != old_total:
                        progress_bar.total += total - old_total
                        progress_bar.refresh()
            elif event == "epoch":
                current = payload.get("current")
                if current is None:
                    progress_bar.update(1)
                    continue

                delta = current - last_epochs.get(job_id, 0)
                if delta > 0:
                    progress_bar.update(delta)
                    last_epochs[job_id] = current
            elif event == "done":
                total = total_epochs_by_job.get(job_id, epochs or EPOCHS)
                remaining = total - last_epochs.get(job_id, 0)
                if remaining > 0:
                    progress_bar.update(remaining)
                    last_epochs[job_id] = total
                active_jobs.discard(job_id)
            elif event == "message":
                tqdm.write(f"{job_id}: {payload.get('text', '')}")

            if active_jobs:
                progress_bar.set_postfix_str("active: " + ", ".join(sorted(active_jobs)))
            else:
                progress_bar.set_postfix_str("")

    def run_job(job):
        index, _version, _scale, model_name, run_name = job
        progress_queue.put(("message", run_name, {"text": f"[{index}/{len(planned)}] start {model_name}"}))
        train_model(
            yaml_path=yaml_path,
            model_name=model_name,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            progress_queue=progress_queue,
            progress_job_id=run_name,
            quiet=True,
        )
        return model_name

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(run_job, job): job for job in jobs}
        progress_bar = tqdm(
            total=(epochs or EPOCHS) * len(jobs),
            desc="Sweep epochs",
            unit="epoch",
            dynamic_ncols=True,
        )
        last_epochs = {job[4]: 0 for job in jobs}
        total_epochs_by_job = {job[4]: epochs or EPOCHS for job in jobs}
        active_jobs = set()
        pending = set(future_to_job)

        try:
            while pending:
                drain_progress(progress_bar, last_epochs, total_epochs_by_job, active_jobs)
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    _index, _version, _scale, model_name, run_name = future_to_job[future]
                    try:
                        future.result()
                        completed += 1
                        drain_progress(progress_bar, last_epochs, total_epochs_by_job, active_jobs)
                        tqdm.write(f"Completed [{completed}/{len(jobs)}]: {model_name} -> {run_name}")
                    except Exception as exc:
                        failures.append((model_name, str(exc)))
                        tqdm.write(f"Failed: {model_name}: {exc}")
        finally:
            drain_progress(progress_bar, last_epochs, total_epochs_by_job, active_jobs)
            progress_bar.close()

    print("\n" + "=" * 70)
    print("Sweep complete")
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
            versions=SWEEP_VERSIONS,
            scales=SWEEP_SCALES,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            device=DEVICE,
            skip_missing_local=SKIP_MISSING_LOCAL,
            parallel=SWEEP_PARALLEL,
        )
        return

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
