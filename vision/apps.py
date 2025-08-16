from django.apps import AppConfig


from django.apps import AppConfig

class VisionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "vision"

    def ready(self):
        # Preload models to avoid first-hit cold start
        from .ai import _ensure_sd, _ensure_resnet, _ensure_yolo
        try:
            _ensure_resnet()
            _ensure_yolo()
            _ensure_sd()
        except Exception:
            # Don’t crash server on environments without weights yet
            pass

