from django import forms

class GenerateForm(forms.Form):
    prompt = forms.CharField(widget=forms.Textarea(attrs={"rows":3}), label="Text prompt")

class ClassifyForm(forms.Form):
    image = forms.ImageField()

class DetectForm(forms.Form):
    image = forms.ImageField()



class TrainingForm(forms.Form):
    job_type = forms.ChoiceField(choices=[("classify","Classification"),("detect","Object Detection (YOLOv8)")])
    dataset_zip = forms.FileField(required=False, help_text="ZIP; classification or YOLO format")
    dataset_path = forms.CharField(required=False, help_text="Existing server path (optional if ZIP used)")
    epochs = forms.IntegerField(min_value=1, initial=5)
    batch_size = forms.IntegerField(min_value=1, initial=8)
    image_size = forms.IntegerField(min_value=64, initial=256)
