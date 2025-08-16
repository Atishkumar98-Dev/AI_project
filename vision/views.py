from django.shortcuts import render, redirect
from django.views import View
from django.core.files.base import ContentFile
from django.utils import timezone
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from django.views import View
from django.shortcuts import render, redirect
from django.http import JsonResponse, Http404
from django.utils import timezone
from .forms import TrainingForm
from .models import TrainingJob
from .training import run_training
import threading

from .forms import GenerateForm, ClassifyForm, DetectForm
from .models import GeneratedImage, UploadedImage
from .serializers import GeneratedImageSerializer, UploadedImageSerializer
from .ai import generate_image, classify_image, detect_objects
from PIL import Image
from io import BytesIO
from pathlib import Path

class HomeView(View):
    template_name = "index.html"
    
    def get(self, request):
        return render(request, self.template_name, {
            "gen_form": GenerateForm(),
            "cls_form": ClassifyForm(),
            "det_form": DetectForm(),
        })

    def post(self, request):
        context = {
            "gen_form": GenerateForm(),
            "cls_form": ClassifyForm(),
            "det_form": DetectForm(),
        }
        

        if "generate_submit" in request.POST:
            
            gen_form = GenerateForm(request.POST)
            context["gen_form"] = gen_form
            if gen_form.is_valid():
                prompt = gen_form.cleaned_data["prompt"]
                # img = generate_image(prompt)
                hq = bool(request.POST.get("hq"))
                # fast: 256px/1 step; HQ: 512px/2–4 steps
                img = generate_image(
                    prompt,
                    height=512 if hq else 256,
                    width=512 if hq else 256,
                    num_inference_steps=3 if hq else 1,
                )
                # Save to model
                buf = BytesIO()
                img.save(buf, format="PNG")
                instance = GeneratedImage(prompt=prompt)
                filename = f"gen_{timezone.now().strftime('%Y%m%d_%H%M%S')}.png"
                instance.image.save(filename, ContentFile(buf.getvalue()))
                instance.save()
                context["generated"] = instance

        elif "classify_submit" in request.POST:
            cls_form = ClassifyForm(request.POST, request.FILES)
            context["cls_form"] = cls_form
            if cls_form.is_valid():
                uploaded = UploadedImage(task="classify", image=cls_form.cleaned_data["image"])
                uploaded.save()
                pil = Image.open(uploaded.image).convert("RGB")
                result = classify_image(pil)
                uploaded.result_json = {"topk": result}
                uploaded.save()
                context["classified"] = uploaded

        elif "detect_submit" in request.POST:
            det_form = DetectForm(request.POST, request.FILES)
            context["det_form"] = det_form
            if det_form.is_valid():
                uploaded = UploadedImage(task="detect", image=det_form.cleaned_data["image"])
                uploaded.save()
                pil = Image.open(uploaded.image).convert("RGB")
                # save annotated output next to upload
                annotated_name = Path(uploaded.image.name).with_suffix("").name + "_det.png"
                annotated_rel = Path("uploads") / annotated_name
                annotated_abs = Path(uploaded.image.storage.location) / annotated_rel
                annotated_abs.parent.mkdir(parents=True, exist_ok=True)
                det = detect_objects(pil, save_annotated_path=annotated_abs)
                uploaded.result_json = {"detections": det, "annotated": str(annotated_rel)}
                uploaded.save()
                context["detected"] = uploaded

        return render(request, self.template_name, context)


# -------- REST API (JSON) ----------

class GenerateAPI(APIView):
    def post(self, request):
        prompt = request.data.get("prompt")
        if not prompt:
            return Response({"detail": "Missing prompt"}, status=status.HTTP_400_BAD_REQUEST)
        img = generate_image(prompt)
        buf = BytesIO()
        img.save(buf, format="PNG")

        instance = GeneratedImage(prompt=prompt)
        filename = f"gen_{timezone.now().strftime('%Y%m%d_%H%M%S')}.png"
        instance.image.save(filename, ContentFile(buf.getvalue()))
        instance.save()

        return Response(GeneratedImageSerializer(instance).data, status=status.HTTP_201_CREATED)


class ClassifyAPI(APIView):
    def post(self, request):
        f = request.FILES.get("image")
        if not f:
            return Response({"detail": "Missing image file"}, status=status.HTTP_400_BAD_REQUEST)
        up = UploadedImage(task="classify", image=f)
        up.save()
        pil = Image.open(up.image).convert("RGB")
        result = classify_image(pil)
        up.result_json = {"topk": result}
        up.save()
        return Response(UploadedImageSerializer(up).data)


class DetectAPI(APIView):
    def post(self, request):
        f = request.FILES.get("image")
        if not f:
            return Response({"detail": "Missing image file"}, status=status.HTTP_400_BAD_REQUEST)
        up = UploadedImage(task="detect", image=f)
        up.save()
        pil = Image.open(up.image).convert("RGB")

        from pathlib import Path
        annotated_name = Path(up.image.name).with_suffix("").name + "_det.png"
        annotated_rel = Path("uploads") / annotated_name
        annotated_abs = Path(up.image.storage.location) / annotated_rel
        annotated_abs.parent.mkdir(parents=True, exist_ok=True)

        det = detect_objects(pil, save_annotated_path=annotated_abs)
        up.result_json = {"detections": det, "annotated": str(annotated_rel)}
        up.save()
        return Response(UploadedImageSerializer(up).data)


class TrainCreateView(View):
    template_name = "vision/train.html"

    def get(self, request):
        return render(request, self.template_name, {"form": TrainingForm(), "jobs": TrainingJob.objects.order_by("-created_at")[:20]})

    def post(self, request):
        form = TrainingForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, self.template_name, {"form": form, "jobs": TrainingJob.objects.order_by("-created_at")[:20]})

        job = TrainingJob.objects.create(
            job_type=form.cleaned_data["job_type"],
            dataset_zip=form.cleaned_data.get("dataset_zip"),
            dataset_path=form.cleaned_data.get("dataset_path") or "",
            hyperparams={
                "epochs": form.cleaned_data["epochs"],
                "batch_size": form.cleaned_data["batch_size"],
                "image_size": form.cleaned_data["image_size"],
            },
            status="queued",
        )
        # kick off the background thread
        threading.Thread(target=run_training, args=(job.id,), daemon=True).start()

        return redirect("train")

class TrainStatusAPI(View):
    def get(self, request, job_id: int):
        try:
            job = TrainingJob.objects.get(id=job_id)
        except TrainingJob.DoesNotExist:
            raise Http404("Job not found")
        return JsonResponse({
            "id": job.id,
            "status": job.status,
            "metrics": job.metrics,
            "output_path": job.output_path,
            "log": job.log[-5000:],  # tail
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        })
