
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
import os
from .predict import classify_image
from django.shortcuts import render

def index(request):
    return render(request, "index.html")

# Scrap classifier : Takes image of scrap : predicted class and accuracy
@api_view(['POST'])
@permission_classes([AllowAny])
def classify_image_view(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    image = request.FILES['image']  # Fetch the image from the request

    try:
        # Save the image and get the path
        image_path = default_storage.save(image.name, image)
        
        # Get the full path to the image
        full_image_path = os.path.join(default_storage.location, image_path)
        
        # Call your classification function
        result = classify_image(full_image_path)  # Adjust based on your function
        
        # Delete the image after processing
        if os.path.exists(full_image_path):
            os.remove(full_image_path)
        
        return Response({'classification': result}, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
