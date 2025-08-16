from django import forms

class GenerateForm(forms.Form):
    prompt = forms.CharField(widget=forms.Textarea(attrs={"rows":3}), label="Text prompt")

class ClassifyForm(forms.Form):
    image = forms.ImageField()

class DetectForm(forms.Form):
    image = forms.ImageField()
