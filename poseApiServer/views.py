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
import sys
import queue
import threading
import poseApiServer.modelServable as ms
# from poseApiServer.modelServable import mxNetModel

sys.path.append("/home/jit2307/AlphaPose-Mxnet")
from Server_API import PoseAPI

Inp_Q = queue.Queue(1000)
Out_Q = queue.Queue(1000)

A = PoseAPI(Inp_Q)
A.run()
Getter= threading.Thread(target=A.out, args=(Out_Q,))

class getPrediction(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def get(self, request):
        if request.method == 'POST':
            inputImage = request.get()
            print(request)
            dataToGet = ms.mxNetModel(inputImage,A,Inp_Q,Out_Q)
            return JsonResponse(
                dataToGet,
                safe=False,content_type='application/json')

