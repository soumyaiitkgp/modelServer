from django.conf.urls import url, include
from rest_framework.authtoken import views
from poseApiServer.views import *

app_name = 'poseApiServer'

urlpatterns = [
    url(r'pred/', getPrediction.as_view()),
]
