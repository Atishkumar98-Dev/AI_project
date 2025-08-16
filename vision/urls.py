from django.urls import path
from .views import HomeView, GenerateAPI, ClassifyAPI, DetectAPI
from .views import TrainCreateView, TrainStatusAPI


urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("api/generate/", GenerateAPI.as_view(), name="api-generate"),
    path("api/classify/", ClassifyAPI.as_view(), name="api-classify"),
    path("api/detect/", DetectAPI.as_view(), name="api-detect"),
    path("train/", TrainCreateView.as_view(), name="train"),
    path("train/<int:job_id>/status/", TrainStatusAPI.as_view(), name="train-status"),
]
