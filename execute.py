import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import itertools
import pathlib
import warnings
import os
import random
import time
import gc
from IPython.display import Markdown, display
from PIL import Image
from random import randint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import keras
from tensorflow import keras
from keras import Sequential
from keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout,SeparableConv2D, Activation, BatchNormalization, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

times = {'1-00': 0, '1-05': 1, '1-10': 2, '1-15': 3, '1-20': 4, '1-25': 5, '1-30': 6, '1-35': 7, '1-40': 8, '1-45': 9,
        '1-50': 10, '1-55': 11, '10-00': 12, '10-05': 13, '10-10': 14, '10-15': 15, '10-20': 16, '10-25': 17, '10-30': 18, 
        '10-35': 19, '10-40': 20, '10-45': 21, '10-50': 22, '10-55': 23, '11-00': 24, '11-05': 25, '11-10': 26, '11-15': 27, 
        '11-20': 28, '11-25': 29, '11-30': 30, '11-35': 31, '11-40': 32, '11-45': 33, '11-50': 34, '11-55': 35, '12-00': 36, 
        '12-05': 37, '12-10': 38, '12-15': 39, '12-20': 40, '12-25': 41, '12-30': 42, '12-35': 43, '12-40': 44, '12-45': 45,
        '12-50': 46, '12-55': 47, '2-00': 48, '2-05': 49, '2-10': 50, '2-15': 51, '2-20': 52, '2-25': 53, '2-30': 54, '2-35': 55,
        '2-40': 56, '2-45': 57, '2-50': 58, '2-55': 59, '3-00': 60, '3-05': 61, '3-10': 62, '3-15': 63, '3-20': 64, '3-25': 65,
        '3-30': 66, '3-35': 67, '3-40': 68, '3-45': 69, '3-50': 70, '3-55': 71, '4-00': 72, '4-05': 73, '4-10': 74, '4-15': 75,
        '4-20': 76, '4-25': 77, '4-30': 78, '4-35': 79, '4-40': 80, '4-45': 81, '4-50': 82, '4-55': 83, '5-00': 84, '5-05': 85,
        '5-10': 86, '5-15': 87, '5-20': 88, '5-25': 89, '5-30': 90, '5-35': 91, '5-40': 92, '5-45': 93, '5-50': 94, '5-55': 95,
        '6-00': 96, '6-05': 97, '6-10': 98, '6-15': 99, '6-20': 100, '6-25': 101, '6-30': 102, '6-35': 103, '6-40': 104, '6-45': 105,
        '6-50': 106, '6-55': 107, '7-00': 108, '7-05': 109, '7-10': 110, '7-15': 111, '7-20': 112, '7-25': 113, '7-30': 114, '7-35': 115,
        '7-40': 116, '7-45': 117, '7-50': 118, '7-55': 119, '8-00': 120, '8-05': 121, '8-10': 122, '8-15': 123, '8-20': 124, '8-25': 125,
        '8-30': 126, '8-35': 127, '8-40': 128, '8-45': 129, '8-50': 130, '8-55': 131, '9-00': 132, '9-05': 133, '9-10': 134, '9-15': 135,
        '9-20': 136, '9-25': 137, '9-30': 138, '9-35': 139, '9-40': 140, '9-45': 141, '9-50': 142, '9-55': 143}

data_time = {v:k for k,v in times.items()}
print(data_time[90])


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = models.load_model('model.h5')
#print(model.weights[0].shape)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

#for p in dir(model):
#    print(p)


video = cv2.VideoCapture(1)

while True:
    _, frame = video.read()

    #Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')

    #Resizing into dimensions you used while training
    im = im.resize((200,200))
    img_array = np.array(im)

    #Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = np.reshape(100,)

    #Calling the predict function using keras
    prediction = np.argmax(model.predict(img_array), axis=1)#[0][0]
    print(data_time[prediction[0]])
    #print(prediction)

    cv2.imshow("Prediction", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
