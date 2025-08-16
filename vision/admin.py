from django.contrib import admin
from .models import GeneratedImage, UploadedImage

@admin.register(GeneratedImage)
class GenAdmin(admin.ModelAdmin):
    list_display = ("id","prompt","created_at")

@admin.register(UploadedImage)
class UpAdmin(admin.ModelAdmin):
    list_display = ("id","task","created_at")
