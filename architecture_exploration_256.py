import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model
import pandas as pd
import sys
import random
import time
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
# record loss during training

class LossHistory(keras.callbacks.Callback):
    def resetHistory(self,x):
        self.reset = x

    def on_train_begin(self, logs={}):
        if self.reset:
            self.train_losses = []
            self.test_losses = []
            self.train_acc = []
            self.test_acc = []
            self.reset = 0

    def on_epoch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        self.test_losses.append(logs.get('val_loss'))
        self.test_acc.append(logs.get('val_acc'))

history = LossHistory()
#model_name = 'Ensembling_Models/0%/model_103_d80p'
#model_name = 'Saved_previous_models/model_102_d80p'
model_name = 'models/model_91_d80p'


dataAugumentation = False
reset_model = False
plot_only = False
data = 'data/data_with_hv'
AdamOptimzer = False
RMSprop = False
predict = False
r = False
k_fold = False
num_of_folds = 10
plot_results = False
include_nih_data = False
nihd = 'D:/xraydata/nih/'
diseases = ("Pleural_Thickening",\
    "Pneumothorax",\
    "Consolidation",\
    "Nodule",\
    "No Finding",\
    "Effusion",\
    "Cardiomegaly",\
    "Mass",\
    "Pneumonia",\
    "Atelectasis",\
    "Emphysema",\
    "Infiltration",\
    "Hernia",\
    "Fibrosis",\
    "Edema")
d2idex = {}
for i,disease in enumerate(diseases):
    d2idex[disease] = i

class_weights = {0:1, 1:10}

#def weighted_binary_crossentropy(y_true, y_pred):
#    loss = -(y_true*K.log(y_pred)*2 + (1-y_true)*K.log(1-y_pred)*1.0)
#    return K.mean(loss, axis=-1)

if not os.path.isfile(model_name + '.h5py') or reset_model:
    history.resetHistory(True)
    model = keras.models.Sequential()
    if model_name == 'models/model_1' or model_name == 'models/model_1_d50p' or \
        model_name == 'models/model_1_d80p':
        model.add(Conv2D(1,(3,3), padding='same', input_shape=(256,256,1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3),padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_1_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_1_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_2':
        model.add(Conv2D(2,(3,3), padding='same', input_shape=(256,256,1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3),padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_2_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_2_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_3':
        model.add(Conv2D(4,(3,3), padding='same', input_shape=(256,256,1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3),padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_3_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_3_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_4' or model_name == 'models/model_4_d50p' or \
        model_name == 'models/model_4_d80p':
        model.add(Conv2D(8,(3,3), padding='same', input_shape=(256,256,1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_4_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_4_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_5' or model_name == 'models/model_5_d50p' or \
                    model_name == 'models/model_5_d80p':
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_5_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_5_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_6' or model_name == 'models/model_6_d50p' or \
                    model_name == 'models/model_6_d80p':
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_6_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_6_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_7' or model_name == 'models/model_7_d50p' or \
                    model_name == 'models/model_7_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_6_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_6_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_8' or model_name == 'models/model_8_d50p' or \
                    model_name == 'models/model_8_d80p':
        model.add(Conv2D(128, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_8_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_8_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_11' or model_name == 'models/model_11_d50p' or \
                    model_name == 'models/model_11_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_11_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_11_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_12' or model_name == 'models/model_12_d50p' or \
                    model_name == 'models/model_12_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_12_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_12_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_13' or model_name == 'models/model_13_d50p' or \
                    model_name == 'models/model_13_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_13_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_13_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_14' or model_name == 'models/model_14_d50p' or \
                    model_name == 'models/model_14_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_14_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_14_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_15' or model_name == 'models/model_15_d50p' or \
                    model_name == 'models/model_15_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_15_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_15_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_21' or model_name == 'models/model_21_d50p' or \
                    model_name == 'models/model_21_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_21_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_21_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_22' or model_name == 'models/model_22_d50p' or \
                    model_name == 'models/model_22_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_22_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_22_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_23' or model_name == 'models/model_23_d50p' or \
                    model_name == 'models/model_23_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_23_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_23_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_31' or model_name == 'models/model_31_d50p' or \
                    model_name == 'models/model_31_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_31_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_31_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_32' or model_name == 'models/model_32_d50p' or \
                    model_name == 'models/model_32_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_32_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_32_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_41' or model_name == 'models/model_41_d50p' or \
                    model_name == 'models/model_41_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_41_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_41_d80p':
            model.add(Dropout(0.8))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_41_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_41_d80p':
            model.add(Dropout(0.8))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_41_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_41_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_51' or model_name == 'models/model_51_d50p' or \
                    model_name == 'models/model_51_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_51_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_51_d80p':
            model.add(Dropout(0.8))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_51_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_51_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_52' or model_name == 'models/model_52_d50p' or \
                    model_name == 'models/model_52_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_52_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_52_d80p':
            model.add(Dropout(0.8))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_52_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_52_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_53' or model_name == 'models/model_53_d50p' or \
                    model_name == 'models/model_53_d80p' or model_name == 'models/model_53_d50pa':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_53_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_53_d80p':
            model.add(Dropout(0.8))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_53_d50pa':
            model.add(Dropout(0.5))
        if model_name == 'models/model_53_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_53_d50pa1':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_53_nl_32':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(32, activation=tf.nn.sigmoid))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_61' or model_name == 'models/model_61_d50p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_61_d50p':
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_71' or model_name == 'models/model_71_d50p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_71_d50p':
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_72' or model_name == 'models/model_72_d50p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_72_d50p':
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_81' or model_name == 'models/model_81_d50p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
        if model_name == 'models/model_81_d50p':
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_82' or model_name == 'models/model_82_d50p' or model_name == 'models/model_82_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_82_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_82_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_91' or model_name == 'models/model_91_d80p' or model_name == 'Ensembling_Models/100%/model_weighted_91_d80p' or model_name == 'Ensembling_Models/0%/re_model_91_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same',activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_91_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_91_d80p' or model_name == 'Ensembling_Models/0%/re_model_91_d80p':
            model.add(Dropout(0.8))
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_101' or model_name == 'models/model_101_d50p' or model_name == 'models/model_101_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_101_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_101_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_102' or model_name == 'models/model_102_d50p' or model_name == 'Ensembling_Models/0%/re_model_102_d80p'or model_name == 'Ensembling_Models/0%/model_weighted_102_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_102_d50p':
            model.add(Dropout(0.5))
        #elif model_name == 'models/model_102_d80p':
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_103' or model_name == 'models/model_103_d50p' \
            or model_name == 'models/model_103_d80p' or model_name == 'models/model_103_k_fold' or \
            model_name == 'models/model_103_data_aug_d50p_k_fold' or \
            model_name == 'models/model_103_nih' or model_name == 'models/model_103_nih_d50_nofold':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_103_d50p' or \
                        model_name == 'models/model_103_data_aug_d50p_k_fold' or \
            model_name == 'models/model_103_nih_d50_nofold':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_103_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_103_drop50_on_last3' or model_name == 'models/model_103_drop505080':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Dropout(0.5))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Dropout(0.5))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_103_drop505080':
            model.add(Dropout(0.8))
        else:
            model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_106' or model_name == 'models/model_106_d50p' or model_name == 'models/model_106_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(4096, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_104_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_104_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'models/model_105' or model_name == 'models/model_105_k_fold' or model_name == 'models/model_105_d50p' or model_name == 'models/model_105_d80p' or model_name == 'models/model_105_adam' or model_name == 'models/model_105_rms_prop' or model_name == 'models/model_105_k_fold':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_105_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_105_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

    elif model_name == 'vgg16' or model_name == 'vgg16_d50p' or model_name == 'vgg16_d80p' or model_name == 'vgg16_dat_aug':
        #model = VGG16(weights='imagenet', include_top=True)
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'vgg16_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'vgg16_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'vgg19' or model_name == 'vgg19_d50p' or model_name == 'vgg19_d80p':
        #model = VGG16(weights='imagenet', include_top=True)
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'vgg19_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'vgg19_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    else:
        print ("Model not defined")
        exit()
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(64, activation=tf.nn.relu))
        # model.add(Dense(1, activation=tf.nn.sigmoid))

    print(model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    elif RMSprop:
        model.compile(optimizer=keras.optimizers.RMSprop(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=keras.optimizers.SGD(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

else:
    history.resetHistory(False)
    print("Loading model from %s" %(model_name + '.h5py'))
    model = load_model(model_name + '.h5py')
    print (model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    elif RMSprop:
        model.compile(optimizer=keras.optimizers.RMSprop(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=keras.optimizers.SGD(decay=0.1),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    with open(model_name + '.aux_data', 'rb') as fin:
        test_losses, train_losses, test_acc, train_acc = pickle.load(fin)
    history.test_losses = test_losses
    history.train_losses = train_losses
    history.test_acc = test_acc
    history.train_acc = train_acc
    print (len(history.test_losses))

# plot the model
plot_model(model, to_file=model_name + '.png')

# normalization constant
norm_const = np.array([255.0])
norm_const = norm_const.astype('float16')

start = time.time()
if not predict:
    with open(data, 'rb') as fin:
        x_train, y_train = pickle.load(fin)
    if include_nih_data:
        x_0 = []
        y_0 = []
        x_1 = []
        y_1 = []
        for i,disease in enumerate(diseases):
            with open(nihd+disease+".data", 'rb') as fin:
                x_tmp = pickle.load(fin)
                if disease == "Pneumonia" or disease == 'Pneumothorax':
                    if len(x_0) == 0:
                        x_0 = x_tmp
                        y_0 = np.zeros((x_tmp.shape[0],1))
                        y_0.fill(d2idex[disease])
                    else:
                        x_0 = np.concatenate((x_0,x_tmp))
                        y_tmp = np.zeros((x_tmp.shape[0], 1))
                        y_tmp.fill(d2idex[disease])
                        y_0 = np.concatenate((y_0,y_tmp))
                else:
                    if len(x_1) == 0:
                        x_1 = x_tmp
                        y_1 = np.zeros((x_tmp.shape[0], 1))
                        y_1.fill(d2idex[disease])
                    else:
                        x_1 = np.concatenate((x_1,x_tmp))
                        y_tmp = np.zeros((x_tmp.shape[0], 1))
                        y_tmp.fill(d2idex[disease])
                        y_1 = np.concatenate((y_1,y_tmp))

        print (x_0.shape)
        print (x_1.shape)
        print (y_0.shape)
        print (y_1.shape)

        y_0.fill(True)
        y_1.fill(False)
        indices = np.arange(x_1.shape[0])
        np.random.shuffle(indices)

        print ("Prior to adding nih images")
        print(x_train.shape)
        print(y_train.shape)

        x_all = np.concatenate((x_train,x_0,x_1[indices[0:x_0.shape[0]]]))
        y_all = np.concatenate((y_train,y_0,y_1[indices[0:x_0.shape[0]]]))
        del x_0,y_0,x_1,y_1

        # shuffle again
        indices = np.arange(x_all.shape[0])
        np.random.shuffle(indices)
        x_train = x_all[indices]
        y_train = y_all[indices]
        del x_all, y_all

        print ("After adding nih images")
        print (x_train.shape)
        print (y_train.shape)

        if False:
            for k in range(4):
                fig, ax = plt.subplots(nrows=2, ncols=2)
                for i in range(2):
                    for j in range(2):
                        if i%2==0:
                            index = random.randint(0, x_0.shape[0])
                            img = x_0[index, :, :, 0]
                            disease = diseases[int(y_0[index,0])]
                        else:
                            index = random.randint(0, x_1.shape[0])
                            img = x_1[index, :, :, 0]
                            disease = diseases[int(y_1[index, 0])]
                        ax[i][j].imshow(img, cmap='gray')
                        ax[i][j].axis('off')
                        ax[i][j].text(0, 0, disease)

                plt.show()
        #exit()

    if dataAugumentation:
        generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=False,
            fill_mode='nearest',
            brightness_range=(0,0.4)
        )
        generator.fit(x_train)
        print (x_train.shape[0])
        for x_aug, y_aug in generator.flow(x_train,y_train, batch_size=int(x_train.shape[0]/4)):
            x_train = np.concatenate((x_train,x_aug), axis=0)
            y_train = np.concatenate((y_train,y_aug), axis=0)
            break

        x_train = x_train.astype('float16')
        x_train = x_train / norm_const
        if k_fold:
            foldsize = int(x_train.shape[0]) / num_of_folds
            foldsize = int(foldsize)
            for e in range(4):
                print("Running super epoch %s" % (e))
                for i in range(num_of_folds):
                    print("Running fold %i" % (i))
                    if i == 0:
                        x_batch = x_train[:(num_of_folds - 1) * foldsize]
                        y_batch = y_train[:(num_of_folds - 1) * foldsize]
                        x_val = x_train[(num_of_folds - 1) * foldsize:]
                        y_val = y_train[(num_of_folds - 1) * foldsize:]
                    elif i == 1:
                        x_batch = x_train[foldsize:]
                        y_batch = y_train[foldsize:]
                        x_val = x_train[:foldsize]
                        y_val = y_train[:foldsize]
                    else:
                        x_batch = np.concatenate((x_train[i * foldsize:], x_train[:(i - 1) * foldsize]), axis=0)
                        y_batch = np.concatenate((y_train[i * foldsize:], y_train[:(i - 1) * foldsize]), axis=0)
                        x_val = x_train[(i - 1) * foldsize:i * foldsize]
                        y_val = y_train[(i - 1) * foldsize:i * foldsize]
                    model.fit(x_batch, y_batch, epochs=1,
                              validation_data=(x_val, y_val),
                              callbacks=[history])
                    model.save(model_name + '.h5py')
                    with open(model_name + '.aux_data', 'wb') as fout:
                        pickle.dump((history.test_losses, history.train_losses, history.test_acc, history.train_acc),
                                    fout)
#        val_generator = generator.flow(x_train,y_train,subset='validation', batch_size=32)
#        model.fit_generator(generator=train_generator,
#                            steps_per_epoch=int(len(x_train)*0.8/32),
#                            validation_data=val_generator,
#                            validation_steps=int(len(x_train)*0.2/32),
#                            epochs=40,
#                            callbacks=[history,
#                                keras.callbacks.EarlyStopping(monitor='acc', patience=40)])
    else:

        x_train = x_train/norm_const
        if k_fold:
            foldsize = int(x_train.shape[0])/num_of_folds
            foldsize = int(foldsize)
            for e in range(4):
                print ("Running super epoch %s" %(e))
                for i in range(num_of_folds):
                    print("Running fold %i" % (i))
                    if i == 0:
                        x_batch = x_train[:(num_of_folds-1)*foldsize]
                        y_batch = y_train[:(num_of_folds-1)*foldsize]
                        x_val = x_train[(num_of_folds-1)*foldsize:]
                        y_val = y_train[(num_of_folds-1)*foldsize:]
                    elif i == 1:
                        x_batch = x_train[foldsize:]
                        y_batch = y_train[foldsize:]
                        x_val = x_train[:foldsize]
                        y_val = y_train[:foldsize]
                    else:
                        x_batch = np.concatenate((x_train[i*foldsize:], x_train[:(i-1)*foldsize]), axis = 0)
                        y_batch = np.concatenate((y_train[i*foldsize:], y_train[:(i-1)*foldsize]), axis = 0)
                        x_val = x_train[(i-1)*foldsize:i*foldsize]
                        y_val = y_train[(i - 1) * foldsize:i*foldsize]
                    model.fit(x_batch, y_batch, epochs = 1,
                              validation_data=(x_val,y_val),
                              callbacks=[history])
                    model.save(model_name + '.h5py')
                    with open(model_name + '.aux_data', 'wb') as fout:
                        pickle.dump((history.test_losses, history.train_losses, history.test_acc, history.train_acc),
                                    fout)
        if not k_fold:
            print (x_train.dtype)
            #print(y_train.sum())
            #print(x_train.shape)
            #print(y_train.shape)
            model.fit(x_train, y_train, batch_size=16, epochs=40,
                      validation_split=0.2,
                      callbacks=[history,
                                 keras.callbacks.EarlyStopping(monitor='acc', patience=4)])

            # Saving history into file
            model.save(model_name + '.h5py')
            with open(model_name + '.aux_data', 'wb') as fout:
                pickle.dump((history.test_losses,history.train_losses,history.test_acc,history.train_acc), fout)

y_test = []
x_test = []
if predict:
    with open('data/rawdata_exp_256_test.dat', 'rb') as fin:
        test = pickle.load(fin)
    #print(test)
    #print(test.shape)
    #exit()
    for i in range(test['patientId'].count()):
        y_test.append([test.iloc[i]['Target']])
        x_test.append([test.iloc[i]['pixel_data']])
    sum = 0
    for i in y_test:
        if i == [1]:
            sum = sum + 1
    print(str(sum))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], 256, 256, 1))
    print(x_test.shape)
    x_test = x_test/norm_const
    if r:
        y_pred = []
        for i in range(x_test.shape[0]):
            a = random.randint(0,1)
            y_pred.append([a])
        y_pred = np.array(y_pred)
    else:
        y_hat = model.predict(x_test)
        print (y_hat)
        y_pred = y_hat > 0.5

        if plot_results:
            for k in range(4):
                fig, ax = plt.subplots(nrows=2, ncols=2)
                for i in range(2):
                    for j in range(2):
                        index = random.randint(0, y_pred.shape[0])
                        img = x_test[index, :, :, 0]
                        ax[i][j].imshow(img, cmap='gray')
                        ax[i][j].axis('off')
                        ax[i][j].text(0, 0, "P="+str([y_pred[index][0]]) + '   R='+str([y_test[index][0]]))
                plt.show()

    # a = tf.cast(tf.argmax(y_pred, 1), tf.float32)
    # b = tf.cast(tf.argmax(y_test, 1), tf.float32)
    #
    # auc = tf.metrics.auc(a,b)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())  # try commenting this line and you'll get the error
    # train_auc = sess.run(auc)
    #
    # print(train_auc)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    df_cm = pd.DataFrame(cm, index= ['True 0','True 1'], columns= ['Predicted 0','Predicted 1'])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt= 'g')
    acc = (cm[0,0] + cm[1,1])/(cm[1,0] + cm[0,1] + cm[0,0] + cm[1,1])
    print("Accuracy:   " + str(acc))
    rec = cm[1,1]/(cm[1,1] + cm[1,0])
    pre = cm[1,1]/(cm[1,1] + cm[0,1])
    f1 = 2 * ((pre*rec)/(pre+rec))
    print('Recall:    ' + str(rec))
    print('Precision:    ' + str(pre))
    print('F1 score:    ' + str(f1))
end = time.time()
print(end - start)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)

# Print ROC curve
plt.plot(fpr,tpr)
plt.show()

# Print AUC
auc = metrics.auc(tpr,fpr)
print('AUC:', auc)

def line_plot(n, bins, _):
    bins1 = []
    length = len(bins) - 1
    for i in range(length):
        a = (bins[i]+bins[i+1])/2
        bins1.append(a)
    plt.plot(bins1, n, color = 'orangered')

id_p = []
id_n = []
y_hat_p = []
y_hat_n = []

for id,label in enumerate(y_test):
    if label[0] == 1:
        id_p.append(id)
        y_hat_p.append(y_hat[id])
    else:
        id_n.append(id)
        y_hat_n.append(y_hat[id])

plt.scatter(id_p, y_hat_p, marker='x', c='red')
plt.show()
plt.scatter(id_n, y_hat_n, marker='x', c='green')
plt.show()

## plot the loss/acc
fig,ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(history.train_losses, color='r')
ax[0].plot(history.test_losses, color='b')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epocs')
ax[0].set_title("Loss vs epocs, train(Red)")
ax[1].plot(history.train_acc, color='r')
ax[1].plot(history.test_acc, color='b')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('epocs')
ax[1].set_title("Accuracy vs epocs, train(Red)")

plt.show()