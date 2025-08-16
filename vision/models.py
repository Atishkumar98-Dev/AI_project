from django.db import models

class GeneratedImage(models.Model):
    prompt = models.TextField()
    image = models.ImageField(upload_to="generated/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Gen:{self.id} ({self.created_at:%Y-%m-%d %H:%M})"


class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    result_json = models.JSONField(default=dict, blank=True)
    task = models.CharField(
        max_length=32,
        choices=[("classify", "classify"), ("detect", "detect")],
        default="classify",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload:{self.id} {self.task}"


class TrainingJob(models.Model):
    JOB_TYPES = (("classify", "Classification"), ("detect", "Object Detection"))
    STATUS = (
        ("queued", "queued"), ("running", "running"),
        ("done", "done"), ("error", "error")
    )

    job_type = models.CharField(max_length=16, choices=JOB_TYPES)
    dataset_zip = models.FileField(upload_to="datasets/zips/", null=True, blank=True)
    dataset_path = models.CharField(max_length=512, blank=True, help_text="Server path (optional if ZIP provided)")
    status = models.CharField(max_length=16, choices=STATUS, default="queued")
    hyperparams = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    output_path = models.CharField(max_length=512, blank=True)
    log = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.id}:{self.job_type}:{self.status}"