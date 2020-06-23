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

import autoencoder_mayank.test1 as mayank
from captureImage.frameAcq import *

class captureImage(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        requestData = request.data
        dataToSend = {}
        dataToSend['framePaths'] = []
        framePaths = acquireFrames(requestData)

        for frame in framePaths:
            tempFrame = frame
            if(frame['camera'] == '4'):

                path = frame['qualityProjectPath']+frame['relativePath']
                img = cv2.imread(path)
                answer = mayank.input(img)
                print(answer)
                tempFrame['autoencoder_mayank'] = {
                    'ssim' : answer,
                    'okNg' : "NG"
                }
                dataToSend['framePaths'].append(tempFrame)
            else:
                dataToSend['framePaths'].append(tempFrame)

        # dataToSend = {
        #     "framePaths" : framePaths
        # }

        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
