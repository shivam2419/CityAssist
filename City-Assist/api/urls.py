from django.urls import path
from .views import classify_image_view, index

urlpatterns = [
    path('predict-road/', classify_image_view, name='classify-image'),
    path('', index, name='index'),
]
