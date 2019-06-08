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

from nltkServer.commentClassification import classifyComments

class getPrediction(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        print("bodyyy",request.data)

        inputImage = [
         "qqwsqw",
         "None",
         "1.plant closed",
         "2.tea time",
         "3.dinner time",
         "1.plant closed",
         "2.dinner time",
         "PLAN NOT UPDATE FOR 28.05.2019",
         "1.manpower shifted to another line.",
         "2.tool problem",
         "1. plant closed",
         "2.dinner time",
         "1.plant closed",
         "2.tea time",
         "1.plant closed",
         "2.lunch time",
         "3.tea time",
         "4.dinner time",
         "1.plant closed",
         "2.lunch time",
         "3.dinner time",
         "1.plant closed",
         "2.lunch time",
         "3.dinner time",
         "1.plant closed"
        ]

        dataToSend = classifyComments(request.data)
        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
