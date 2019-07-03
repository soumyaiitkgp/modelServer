import os
import os.path as osp
import time
import gluoncv as gcv
import mxnet as mx
import cv2
import threading

# numw = 1
# workers = []
# getters = []
# for i in range(numw):
#     workers.append(PoseAPI())
# for i in range(numw):
#     getters.append(threading.Thread(target=workers[i].out, args=(Out_Q,5) ) )
# for i in range(numw):
#     workers[i].run()
#     getters[i].start()

def mxNetModel(inputData,A):
    A.input(inputData,Inp_Q)
    size = len(Inp_Q)
    outputData = []
    while size > 0:
        if not Out_Q.empty():
            outputData.append(Out_Q.get())
            size = size - 1;
        else:
            time.sleep(0.1);
    return outputData
