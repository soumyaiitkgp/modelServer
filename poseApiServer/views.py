from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.http import HttpResponse
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
import os
import poseApiServer.modelServable as ms
from django.shortcuts import render
from django.template import Context, loader
# from poseApiServer.modelServable import mxNetModel

sys.path.append("/home/jit2307/AlphaPose-Mxnet")
from Server_API import PoseAPI

def prev_dir(filename):
    a = filename.split("/")
    s = ""
    for i in range (len(a)-1):
        s = s + a[i] + "/"
    return s

scriptpath = os.path.realpath(__file__)
dirpath = prev_dir(scriptpath)

Inp_Q = queue.Queue(1000)
Out_Q = queue.Queue(1000)

A = PoseAPI(Inp_Q)
A.run()
Getter= threading.Thread(target=A.out, args=(Out_Q,))
def getimage(request):
    if request.method == 'GET':
        template = loader.get_template(dirpath+"templates/index.html")
        return HttpResponse(template.render())

class getPrediction(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        inputImage = request.data
        print("hidbghd")
        dataToGet = ms.mxNetModel(inputImage,A,Inp_Q,Out_Q)
        return JsonResponse(
            dataToGet,
            safe=False,content_type='application/json')

