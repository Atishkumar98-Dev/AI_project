import io, os, shutil, zipfile, json, threading, time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from django.conf import settings
from django.utils import timezone

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from ultralytics import YOLO

from .models import TrainingJob

DATASETS_ROOT: Path = settings.DATASETS_ROOT
MODELS_DIR: Path = Path(settings.MEDIA_ROOT) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def _append_log(job: TrainingJob, msg: str):
    job.log += f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    job.save(update_fields=["log"])

def _extract_zip_to(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)

def _resolve_dataset(job: TrainingJob) -> Path:
    """Return a directory path containing the dataset."""
    job_folder = DATASETS_ROOT / f"job_{job.id}"
    job_folder.mkdir(parents=True, exist_ok=True)

    if job.dataset_zip:
        src = Path(job.dataset_zip.path)
        _append_log(job, f"Unpacking ZIP: {src.name}")
        _extract_zip_to(src, job_folder)
        # if the zip had a top-level single folder, descend into it
        items = list(job_folder.iterdir())
        if len(items) == 1 and items[0].is_dir():
            return items[0]
        return job_folder

    if job.dataset_path:
        p = Path(job.dataset_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"dataset_path not found: {p}")
        return p

    raise ValueError("No dataset_zip or dataset_path provided.")

# ---------- Classification training ----------
def train_classifier(job: TrainingJob, ds_root: Path, epochs=5, batch=8, img_size=256) -> Dict[str, Any]:
    _append_log(job, f"Preparing classification dataset at: {ds_root}")

    train_dir = ds_root / "train"
    val_dir = ds_root / "val"
    tfms = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
    }

    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=tfms["train"])
        if val_dir.exists():
            val_ds = datasets.ImageFolder(val_dir, transform=tfms["val"])
        else:
            # 80/20 split from train if no val provided
            n = len(train_ds)
            v = max(1, int(0.2 * n))
            t = n - v
            train_ds, val_ds = random_split(train_ds, [t, v])
    else:
        # assume flat folder: split automatically
        full_ds = datasets.ImageFolder(ds_root, transform=tfms["train"])
        n = len(full_ds)
        v = max(1, int(0.2 * n))
        t = n - v
        train_ds, val_ds = random_split(full_ds, [t, v])
        # switch val transform
        val_ds.dataset.transform = tfms["val"]

    num_classes = len(getattr(getattr(train_ds, "dataset", train_ds), "classes", [])) or len(train_ds.dataset.classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _append_log(job, f"Classes: {num_classes}; device: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_acc = 0.0
    best_path = MODELS_DIR / f"classifier_job{job.id}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(1, total)
        _append_log(job, f"Epoch {epoch}/{epochs} - train_loss={running/max(1,len(train_loader.dataset)):.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": getattr(getattr(train_ds, "dataset", train_ds), "classes", []),
                "img_size": img_size,
            }, best_path.as_posix())

    return {"best_acc": best_acc, "weights": str(best_path)}

# ---------- YOLO detection training ----------
def train_detector(job: TrainingJob, ds_root: Path, epochs=30, batch=16, img_size=640) -> Dict[str, Any]:
    # expects a data.yaml in ds_root (or nested). Try to find one.
    yaml_candidates = list(ds_root.rglob("data.yaml"))
    if not yaml_candidates:
        raise FileNotFoundError("YOLO dataset must include a data.yaml (with train/val paths).")
    data_yaml = yaml_candidates[0]

    device = 0 if torch.cuda.is_available() else "cpu"
    _append_log(job, f"Found data.yaml at {data_yaml}; device={device}")

    model = YOLO("yolov8n.pt")
    res = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        workers=2,
        half=torch.cuda.is_available(),
        verbose=True
    )
    # Ultralytics saves to runs/detect/train*/
    best = Path(res.save_dir) / "weights" / "best.pt"
    out_path = MODELS_DIR / f"detector_job{job.id}_best.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    return {"weights": str(out_path), "run_dir": str(res.save_dir)}

# ---------- Orchestrator ----------
def run_training(job_id: int):
    job = TrainingJob.objects.get(id=job_id)
    try:
        job.status = "running"
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at"])

        ds_root = _resolve_dataset(job)
        hp = {"epochs": job.hyperparams.get("epochs", 5),
              "batch_size": job.hyperparams.get("batch_size", 8),
              "image_size": job.hyperparams.get("image_size", 256)}

        if job.job_type == "classify":
            metrics = train_classifier(job, ds_root, epochs=hp["epochs"], batch=hp["batch_size"], img_size=hp["image_size"])
        elif job.job_type == "detect":
            # often larger size helps detection; allow override via form
            metrics = train_detector(job, ds_root, epochs=hp["epochs"], batch=hp["batch_size"], img_size=max(384, hp["image_size"]))
        else:
            raise ValueError(f"Unsupported job_type: {job.job_type}")

        job.status = "done"
        job.metrics = metrics
        job.output_path = metrics.get("weights", "")
        job.finished_at = timezone.now()
        job.save()

    except Exception as e:
        _append_log(job, f"ERROR: {e}")
        job.status = "error"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at", "log"])
