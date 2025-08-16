import torch
from PIL import Image
from pathlib import Path
import io
import json

# --- Classification (ResNet-50) ---
from torchvision import models, transforms

# --- Detection (YOLOv8) ---
from ultralytics import YOLO

# --- Text-to-Image (Stable Diffusion Turbo) ---
from diffusers import AutoPipelineForText2Image

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy singletons to avoid loading models on every request
_RESNET = None
_YOLO = None
_SD_PIPE = None

# ImageNet normalization
_IMG_TFMS = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Human-readable ImageNet classes (download once if you want; here’s a minimal fallback)
# For complete labels, you can fetch ImageNet class index file; minimal fallback:
_IMAGENET_MINI = {0: "tench", 1: "goldfish", 2: "great white shark"}  # placeholder few
# Better: load full mapping when internet/files available:
# try:
#     import json, requests
#     _IMAGENET_IDX = requests.get("https://.../imagenet_class_index.json").json()
# except:
#     _IMAGENET_IDX = None

from torchvision import models
def _ensure_resnet():
    global _RESNET
    if _RESNET is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval().to(_DEVICE)
        _RESNET = model
    return _RESNET


def _ensure_yolo():
    global _YOLO
    if _YOLO is None:
        # pretrained YOLOv8n general model
        _YOLO = YOLO("yolov8n.pt")
        # YOLO handles device internally; set if available
        if torch.cuda.is_available():
            _YOLO.to("cuda")
    return _YOLO

def _ensure_sd():
    global _SD_PIPE
    if _SD_PIPE is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _SD_PIPE = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype
        )
        # --- speed/memory toggles ---
        try:
            _SD_PIPE.safety_checker = None  # small speed boost
        except Exception:
            pass
        if torch.cuda.is_available():
            _SD_PIPE.enable_xformers_memory_efficient_attention()  # if xformers installed
            _SD_PIPE = _SD_PIPE.to("cuda")
            torch.set_float32_matmul_precision("high")
        else:
            _SD_PIPE.enable_attention_slicing()   # helps CPU RAM; minor speed-up
            _SD_PIPE.enable_vae_slicing()
            # _SD_PIPE.enable_sequential_cpu_offload()  # use if RAM is tight; may trade some speed
    return _SD_PIPE



@torch.inference_mode()
def classify_image(pil_image: Image.Image, topk: int = 5):
    model = _ensure_resnet()
    tensor = _IMG_TFMS(pil_image).unsqueeze(0).to(_DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk_probs, topk_idxs = torch.topk(probs, k=topk)
    topk_probs = topk_probs.tolist()
    topk_idxs = [int(i) for i in topk_idxs.tolist()]
    # Labels: try torchvision metadata (available via weights meta)
    try:
        categories = models.ResNet50_Weights.DEFAULT.meta["categories"]
        labels = [categories[i] for i in topk_idxs]
    except Exception:
        labels = [_IMAGENET_MINI.get(i, f"class_{i}") for i in topk_idxs]
    return [{"label": lab, "prob": float(p)} for lab, p in zip(labels, topk_probs)]


def detect_objects(pil_image: Image.Image, save_annotated_path: Path | None = None):
    yolo = _ensure_yolo()
    # run prediction; returns a list of Results
    device = 0 if torch.cuda.is_available() else "cpu"
    # results = yolo.predict(pil_image, conf=0.25, verbose=False)
    results = yolo.predict(
        pil_image,
        conf=0.30,
        imgsz=512,      # was default 640; 512 is faster
        half=torch.cuda.is_available(),  # half-precision on GPU
        device=device,
        verbose=False
    )
    out = []
    if results:
        res = results[0]
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = [float(x) for x in b.xyxy.squeeze(0).tolist()]
            out.append({"class": res.names[cls_idx], "confidence": conf, "box_xyxy": xyxy})
        if save_annotated_path:
            # Save a plotted image with boxes
            annotated = res.plot()  # ndarray (BGR)
            # Convert BGR to RGB PIL
            from cv2 import cvtColor, COLOR_BGR2RGB
            import numpy as np
            pil_annot = Image.fromarray(cvtColor(annotated, COLOR_BGR2RGB))
            pil_annot.save(save_annotated_path)
    return out


@torch.inference_mode()
def generate_image(prompt: str, height: int = 512, width: int = 512, num_inference_steps: int = 2):
    pipe = _ensure_sd()
    # SD-Turbo works best with very few steps
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,  # CFG off for turbo
    ).images[0]
    return image
