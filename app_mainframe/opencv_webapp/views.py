# Create your views here.
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from .opencv_dtemplate import opencv_dtemplate

def first_view(request):
  return render(request, 'opencv_webapp/first_view.html', {})

def dtemplate(request):
  if request.method == 'POST':
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
      post = form.save(commit=False)
      post.save()

      imageURL = settings.MEDIA_URL + form.instance.document.name
      opencv_dtemplate(settings.MEDIA_ROOT_URL + imageURL)

      return render(request, 'opencv_webapp/dtemplate.html', {'form': form, 'post': post})
  else:
    form = ImageUploadForm()
  return render(request, 'opencv_webapp/dtemplate.html', {'form': form})