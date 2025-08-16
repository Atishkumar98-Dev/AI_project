from rest_framework import serializers
from .models import GeneratedImage, UploadedImage

class GeneratedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedImage
        fields = ["id", "prompt", "image", "created_at"]

class UploadedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = ["id", "image", "result_json", "task", "created_at"]
        