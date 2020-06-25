from django.conf.urls import url, include
from rest_framework.authtoken import views
from captureImage.views import *

app_name = 'Maskrcnn_Soumya'

urlpatterns = [
    url(r'predict', predict.as_view()),
]
