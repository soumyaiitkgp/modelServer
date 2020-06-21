import cv2
import os
from datetime import datetime
from datetime import date

cameraList = [
    ("rtsp://admin:Sahil12051994%40@192.168.1.64:554/Streaming/Channels/301",'1'),
    ("rtsp://admin:Sahil12051994%40@192.168.1.64:554/Streaming/Channels/301",'2'),
    ("rtsp://admin:Sahil12051994%40@192.168.1.64:554/Streaming/Channels/301",'3'),
    ("rtsp://admin:Sahil12051994%40@192.168.1.64:554/Streaming/Channels/301",'4'),
    ("rtsp://admin:Sahil12051994%40@192.168.1.64:554/Streaming/Channels/301",'5'),
]

today = date.today()
d1 = today.strftime("%d-%m-%Y")
qualityProjectPath = "/home/sahil/quality_lockDownBackup/JBMQualityproject"
def acquireFrames(data):
    partId = data['partId']
    setupType = str(data['cameraSetup'])
    tempImagePaths = []
    for cam in cameraList:
        if cam[1] in data['imagesToAcquire'] :
            videoCaptureObject = cv2.VideoCapture(cam[0])
            result = True
            while(result):
                ret,frame = videoCaptureObject.read()

                pathAppend = "/data/qualityFrames/" + d1 + "/" + setupType + "_camera/" + partId + "/" + cam[1] + "/"
                directory = qualityProjectPath + pathAppend

                my_date = datetime.now()
                fileName = my_date.strftime('%Y-%m-%dT%H:%M:%S.%f%z') + ".jpg"

                if not os.path.exists(directory):
                    os.makedirs(directory)

                cv2.imwrite((directory + fileName),frame)
                result = False
                tempImagePaths.append(pathAppend + fileName)
            videoCaptureObject.release()
            cv2.destroyAllWindows()
    print("Captured")
    return tempImagePaths
