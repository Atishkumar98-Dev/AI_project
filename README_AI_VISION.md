# AI Vision (Django) — Generate • Classify • Detect • Train

This project is a **Django** web app that can:
- **Generate** images from text (fast preview with SD-Turbo, high quality with SDXL + Upscaler)
- **Classify** images (ResNet-based, supports your own trained classes like `virat_kohli` vs `other`)
- **Detect** objects (YOLOv8; saves annotated images)
- **Train in bulk** (upload/point to datasets, background training, metrics & logs)

---

## Project Structure

```text
ai_vision/
├─ manage.py
├─ requirements.txt
├─ README.md
├─ ai_vision/
│  ├─ __init__.py
│  ├─ settings.py
│  ├─ urls.py
│  └─ wsgi.py
└─ vision/
   ├─ __init__.py
   ├─ apps.py
   ├─ admin.py
   ├─ models.py
   ├─ ai.py
   ├─ views.py
   ├─ serializers.py
   ├─ forms.py
   ├─ training.py
   ├─ urls.py
   └─ templates/
      └─ vision/
         ├─ index.html
         └─ train.html

```

### What each file/folder does

- `ai_vision/settings.py` — Django config (apps, static/media, DB, templates)
- `ai_vision/urls.py` — URL entry that includes the app routes
- `vision/ai.py` — **All AI logic** (text-to-image pipelines, classifiers, detectors, upscalers)
- `vision/training.py` — **Bulk training**: unzips/validates datasets, trains classifier/YOLO, saves weights
- `vision/views.py` — Web views & REST endpoints for generate/classify/detect/train
- `vision/forms.py` — HTML forms for inputs (prompts, images, training params)
- `vision/models.py` — Models for generated/ uploaded images and training jobs
- `vision/templates/vision/*.html` — Pages (`index.html` home, `train.html` training)
- `requirements.txt` — Python dependencies

---

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver 0.0.0.0:8005
```

Open **http://127.0.0.1:8005/**

---

## Features

### 1) Text → Image
- **Fast preview**: SD-Turbo @ 256–512px, 1–4 steps
- **High quality**: SDXL @ 1024px + optional Refiner + x2/x4 Upscaler
- Controls: **negative prompt**, **steps**, **CFG**, **seed**

### 2) Image Classification
- Built-in ResNet head; **auto-loads latest fine-tuned weights** from `media/models`
- Optional **face crop** via MTCNN for better identity accuracy
- Returns top-k labels & probabilities

### 3) Object Detection
- YOLOv8 (nano by default)
- Saves **annotated** image and returns detection list (class, confidence, box)

### 4) Bulk Training
- Upload a ZIP or point to a server path
- **Classification** format:
  ```
  dataset_root/
    train/classA/*.jpg
    train/classB/*.jpg
    val/classA/*.jpg
    val/classB/*.jpg
  ```
- **Detection** (YOLO) format:
  ```
  dataset_root/
    data.yaml
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt   # YOLO txt labels
    labels/val/*.txt
  ```
- Background thread launches training, saves best weights to `media/models/`

---

## API Endpoints

- `POST /api/generate` — JSON: `{ "prompt": "...", "negative_prompt": "...", "hq": true }` → returns generated image info
- `POST /api/classify` — multipart with `image`
- `POST /api/detect` — multipart with `image`
- `GET  /train/<id>/status/` — job progress, metrics, log tail

---

## ELI5 

**Think of it like a kitchen:**  
- **`ai.py`** is the **chef**. It knows recipes for making pictures (generation), tasting pictures (classification), and finding things in pictures (detection).  
- **`views.py`** is the **waiter**. It takes your order (form/API), asks the chef to cook, and brings back the dish (the result).  
- **`forms.py`** is the **menu**. You tick boxes or type prompts.  
- **`templates/*.html`** is the **table** where your food arrives (web pages).  
- **`training.py`** is a **cooking class** for the chef — it practices with lots of pictures to get better at certain foods (e.g., recognizing Virat Kohli).  
- **`models.py`** is the **notebook** that remembers what was cooked and practiced (DB entries with paths to files).

When you click **Generate**, the waiter runs to the chef with your prompt. The chef follows a recipe (SD-Turbo or SDXL), makes a picture, and the waiter puts it on your table (the page). When you upload a photo to **Classify** or **Detect**, the chef checks the photo with different tools and tells you what it is or where things are.

---

## How the Code Fits Together (short)

- **URLs → Views**: `urls.py` maps `/`, `/api/*`, `/train/*` to functions/classes in `views.py`  
- **Views → AI**: those views call helpers in `ai.py` to do the actual ML work  
- **Training pipeline**: view creates a `TrainingJob`, spawns a thread that runs `training.py:run_training(job_id)`  
- **Artifacts**: images under `media/generated/` or `media/uploads/`, models under `media/models/`  
- **Settings**: `MEDIA_URL`/`MEDIA_ROOT` serve those files during development

---

## Tips

- On CPU: use SD-Turbo for previews; keep sizes ≤ 512px.  
- On GPU: enable xformers, use SDXL HQ.  
- For identity (e.g., Kohli): add **negatives** (e.g., `other/`) and consider **face crop** in classification.

---

## License & Credits

- Diffusers / SDXL / SD-Turbo by Stability AI & Hugging Face community.
- YOLOv8 by Ultralytics.
- Verify licenses for any **datasets** you download.
