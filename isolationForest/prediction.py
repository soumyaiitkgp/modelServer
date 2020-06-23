from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest
import pickle
import cv2
import numpy as np

def get_result(path):
    autoencoder = load_model('C:/Users/mukul singh/Documents/JBM Intern/Server/modelServer/isolationForest/auto_encoder.h5')
    mms = pickle.load(open('C:/Users/mukul singh/Documents/JBM Intern/Server/modelServer/isolationForest/mms.pkl', 'rb'))
    forest = pickle.load(open('C:/Users/mukul singh/Documents/JBM Intern/Server/modelServer/isolationForest/forest.pkl', 'rb'))

    img = cv2.imread(path)
    print(path)
    img = cv2.resize(img, (336, 336))
    test = np.array([img])
    # test = test + 20
    test = test/255
    test = test.astype('float32')
    # print(np.max(test))

    features = autoencoder.predict(test)
    features = np.reshape(features, (features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]))
    scaled_features = mms.transform(features)
    result = forest.predict(scaled_features)
    return(result)
