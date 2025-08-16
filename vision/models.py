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
