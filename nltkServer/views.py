from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.http import JsonResponse

from rest_framework.exceptions import ParseError
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from nltkServer.modelServable import classifyComments

class getPrediction(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    print("asdasdas")
    def get(self, request):

        inputImage = "papapaapapap"
        print(request)
        dataToSend = classifyComments(inputImage)
        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
