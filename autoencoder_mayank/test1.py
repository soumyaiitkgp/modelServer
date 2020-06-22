from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import TensorBoard
from SSIM_PIL import compare_ssim as ssim
from keras import backend as K
import tensorflow as tf
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def autoencoderModel(input_shape):

    input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

    # Encode-----------------------------------------------------------
    x = Conv2D(32, (4, 4), strides=2 , activation='relu', padding='same')(input_img)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    encoded = Conv2D(1, (8, 8), strides=1, padding='same')(x)

    # Decode---------------------------------------------------------------------
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(encoded)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (8, 8), activation='sigmoid', padding='same')(x)
    # ---------------------------------------------------------------------
    model = Model(input_img, decoded)
    return model

def get_output(x_test):
    width = 128
    height = 128
    pixels = width * height * 1 
    x_test = x_test.astype('float32')/255.
    x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))  # adapt this if using `channels_first` image data format
    # print (x_train.shape)
    print (x_test.shape)
    input_shape = x_test.shape[1:]
    autoencoder = autoencoderModel(input_shape)
    autoencoder.load_weights('/home/mayank/djangoJBM/model/autoencoder/weights-improvement-500-0.27.hdf5')
    autoencoder.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss,'accuracy'])
    decoded_imgs = autoencoder.predict(x_test)
    # cv2.imwrite("mayank.jpg",decoded_imgs)
    n = 1 # how many digits we will display
    # plt.figure(figsize=(20, 5), dpi=100)
    # for i in range(n):
        # display original
        # ax = plt.subplot(2, n, i + 1)
        # plt.imshow(x_test[i].reshape(128, 128))
        # plt.gray()
        # ax.get_xaxis().set_visible(True)
        # ax.get_yaxis().set_visible(False)

        # # SSIM Encode
        # ax.set_title("Encode_Image")

    npImg = x_test[0]
    npImg = npImg.reshape((128, 128))
    formatted = (npImg*255 / np.max(npImg)).astype('uint8')
    img = Image.fromarray(formatted)
    img1 = np.asarray(img)
        # cv2.imwrite("img_test"+str(i)+"enc.jpg",img1)

        # display reconstruction
        # ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(128, 128))
        # plt.gray()
        # ax.get_xaxis().set_visible(True)
        # ax.get_yaxis().set_visible(False)

        # SSIM Decoded
        # print("-------------------------------[",i)
    npDecoded = decoded_imgs[0]
    npDecoded = npDecoded.reshape((128, 128))
    formatted2 = (npDecoded *255 / np.max(npDecoded)).astype('uint8')
    decoded = Image.fromarray(formatted2)
    decoded1= np.asarray(decoded)
    value = ssim(img, decoded)
    # print(value)
    return value
    # cv2.imshow("mayank",decoded1)
    # cv2.waitKey(0)


        # value = ssim(img, decoded)

        # label = 'SSIM: {:.3f}'

        # ax.set_title("Decoded_Image")
        # ax.set_xlabel(label.format(value))

    # plt.show()
    # plt.savefig("answ2.jpg")

# img_dir = ("yad/")
# img_files = glob.glob(img_dir + "*.jpg")
# # Setting Image Propertie

# # Load Image
# # AutoEncoder does not have to label data
# x = []
# for i, f in enumerate(img_files):
#     img = cv2.imread(f) 
#     #print(type(img))
     
#     #if type(img) is list:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#           # gray sclae
#     img = cv2.resize(img,(width, height))
#     data = np.asarray(img)
#     x.append(data)
#     #else : 
#        # continue
#     if i % 10 == 0:
#         print(i, "\n", data)

#     print("------------------------------")
#     print(f)
#     print(img)
#     print("------------------------------")
    #     img = img.convert("RGB")
def input(img):
    x=[]
    width = 128
    height = 128
    pixels = width * height * 1  # gray scale

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# x.append(img)     
    img = cv2.resize(img,(width, height))
    data = np.asarray(img)
    x.append(data)
    x = np.array(x)
    # (x_train, x_test) = train_test_split(x, shuffle=False, train_size=0.2, random_state=1)


    # img_list = (x_train, x_test)
    # np.save("./obj_good_ch.npy", img_list)
    print("OK", len(x))
    x_test = x
    imgOut = get_output(x_test)
    # value = ssim(img, imgOut)
    # print("--------------------------------"+value)
    return imgOut

# img = cv2.imread("test.jpg")
# input(img)

# change to float32
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32')/255.
# x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))  # adapt this if using `channels_first` image data format
# print (x_train.shape)
# print (x_test.shape)
# input_shape = x_train.shape[1:]
# autoencoder = autoencoderModel(input_shape)
# autoencoder.load_weights('weights-improvement-310-0.24.hdf5')
# autoencoder.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss,'accuracy'])
# decoded_imgs = autoencoder.predict(x_test)

# n = 7 # how many digits we will display
# plt.figure(figsize=(20, 5), dpi=100)
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(128, 128))
#     plt.gray()
#     ax.get_xaxis().set_visible(True)
#     ax.get_yaxis().set_visible(False)

#     # SSIM Encode
#     ax.set_title("Encode_Image")

#     npImg = x_test[i]
#     npImg = npImg.reshape((128, 128))
#     formatted = (npImg*255 / np.max(npImg)).astype('uint8')
#     img = Image.fromarray(formatted)
#     img1 = np.asarray(img)
#     cv2.imwrite("img_test"+str(i)+"enc.jpg",img1)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(128, 128))
#     plt.gray()
#     ax.get_xaxis().set_visible(True)
#     ax.get_yaxis().set_visible(False)

#     # SSIM Decoded
#     npDecoded = decoded_imgs[i]
#     npDecoded = npDecoded.reshape((128, 128))
#     formatted2 = (npDecoded *255 / np.max(npDecoded)).astype('uint8')
#     decoded = Image.fromarray(formatted2)
#     decoded1= np.asarray(decoded)
#     cv2.imwrite("img_test"+str(i)+"dec.jpg",decoded1)


#     value = ssim(img, decoded)

#     label = 'SSIM: {:.3f}'

#     ax.set_title("Decoded_Image")
#     ax.set_xlabel(label.format(value))

# plt.show()
# plt.savefig("answ2.jpg")