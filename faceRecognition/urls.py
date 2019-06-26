from django.conf.urls import url, include
from rest_framework.authtoken import views
from faceRecognition.views import *

app_name = 'faceRecognition'

urlpatterns = [
    url(r'pred/', getFacePrediction.as_view()),
]
