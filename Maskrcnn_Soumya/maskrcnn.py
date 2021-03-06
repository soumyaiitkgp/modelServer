#!/usr/bin/env python
# coding: utf-8
# ---------------------Author: Soumya Yadav-------------------------
#----------------------Github: https://github.com/soumyaiitkgp------
# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.


import os
import sys
import random
import time
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import Model config
import models.detector as detector

    # Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs//JBM_Test2")  
# Local path to trained weights file
DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, "detector.h5")
# Directory of images to run detection on

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.


class InferenceConfig(detector.DetectorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


# ## Create Model and Load Trained Weights

#print('Loading weights...')
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(DETECTOR_MODEL_PATH, by_name=True)
#print('Weights loaded')

# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

print('loading class_names...')
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'crack','missing hole','profile','slug jam','double punch','small hole','big hole','hole','hole shift']
print('class_names loaded')

# ## Run Object Detection
def detection(path):
    IMAGE_PATH = path
    #file_names = next(os.walk(IMAGE_DIR))[2]
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[0]))
    image = skimage.io.imread(IMAGE_PATH)
    #print(file_names)
    results = model.detect([image],verbose=1)

    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    n = r['class_ids']
    n = n.tolist()
    if n == []:
        out = 'RESULT: The sample is OK'
        #print('RESULT: The sample is OK')
    else:
        #print('RESULT: The sample is not OK')
        out = 'RESULT: The sample is not OK'
    return out
#print('Running detection on the sample...')

#path  = "D:\\Soumya\\modelServer\\Maskrcnn_Soumya\\images\\01.jpg"
#detection(path)

