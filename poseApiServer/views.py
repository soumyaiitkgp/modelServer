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

# from poseApiServer.modelServable import mxNetModel

sys.path.append("/home/jbmai/AlphaPose-Mxnet")
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
<<<<<<< HEAD
        if request.method == 'POST':
            inputImage = request.get();
            print(request)
            dataToGet = mxNetModel(inputImage,A)
            return JsonResponse(
                dataToGet,
                safe=False,content_type='application/json')
=======

        inputImage = "papapaapapap"
        print(request)
        # dataToSend = mxNetModel(inputImage)
        dataToSend = "haha"
        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
>>>>>>> b94f0f93b3a2d8612c25b17bc4d7393f0ed9241c
