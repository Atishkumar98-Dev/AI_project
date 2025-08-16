from django.urls import path
from .views import HomeView, GenerateAPI, ClassifyAPI, DetectAPI

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("api/generate/", GenerateAPI.as_view(), name="api-generate"),
    path("api/classify/", ClassifyAPI.as_view(), name="api-classify"),
    path("api/detect/", DetectAPI.as_view(), name="api-detect"),
]
