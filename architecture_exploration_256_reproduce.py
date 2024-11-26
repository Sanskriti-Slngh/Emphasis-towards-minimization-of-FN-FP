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
model_name = 'models/model_102_d80p'

reset_model = False
train_data = 'data/train_data_with_256'
val_data = 'data/val_data_with_256'
AdamOptimzer = False
RMSprop = False
predict = True
plot_results = False

if not os.path.isfile(model_name + '.h5py') or reset_model:
    history.resetHistory(True)
    model = keras.models.Sequential()
    if model_name == 'models/model_1' or model_name == 'models/model_1_d50p' or model_name == 'models/model_1_d80p':
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
    elif model_name == 'models/model_4' or model_name == 'models/model_4_d50p' or model_name == 'models/model_4_d80p':
        model.add(Conv2D(8,(3,3), padding='same', input_shape=(256,256,1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_4_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_4_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_5' or model_name == 'models/model_5_d50p' or model_name == 'models/model_5_d80p':
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_5_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_5_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_6' or model_name == 'models/model_6_d50p' or model_name == 'models/model_6_d80p':
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_6_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_6_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_7' or model_name == 'models/model_7_d50p' or model_name == 'models/model_7_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        if model_name == 'models/model_6_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_6_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_8' or model_name == 'models/model_8_d50p' or model_name == 'models/model_8_d80p':
        model.add(Conv2D(128, (3, 3), padding='same', input_shape=(256, 256, 1)))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_8_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_8_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_11' or model_name == 'models/model_11_d50p' or model_name == 'models/model_11_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_11_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_11_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_12' or model_name == 'models/model_12_d50p' or model_name == 'models/model_12_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_12_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_12_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_13' or model_name == 'models/model_13_d50p' or model_name == 'models/model_13_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_13_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_13_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_14' or model_name == 'models/model_14_d50p' or model_name == 'models/model_14_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_14_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_14_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_15' or model_name == 'models/model_15_d50p' or model_name == 'models/model_15_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_15_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_15_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_21' or model_name == 'models/model_21_d50p' or model_name == 'models/model_21_d80p':
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
    elif model_name == 'models/model_22' or model_name == 'models/model_22_d50p' or model_name == 'models/model_22_d80p':
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
    elif model_name == 'models/model_23' or model_name == 'models/model_23_d50p' or model_name == 'models/model_23_d80p':
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
    elif model_name == 'models/model_31' or model_name == 'models/model_31_d50p' or model_name == 'models/model_31_d80p':
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
    elif model_name == 'models/model_32' or model_name == 'models/model_32_d50p' or model_name == 'models/model_32_d80p':
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
    elif model_name == 'models/model_41' or model_name == 'models/model_41_d50p' or model_name == 'models/model_41_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_41_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_41_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_51' or model_name == 'models/model_51_d50p' or model_name == 'models/model_51_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_51_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_51_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_52' or model_name == 'models/model_52_d50p' or model_name == 'models/model_52_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Flatten())
        if model_name == 'models/model_52_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_52_d80p':
            model.add(Dropout(0.8))
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_53' or model_name == 'models/model_53_d50p' or model_name == 'models/model_53_d80p':
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(256, 256, 1)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))
        if model_name == 'models/model_53_d50p':
            model.add(Dropout(0.5))
        if model_name == 'models/model_53_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
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
    elif model_name == 'models/model_91' or model_name == 'models/model_91_d50p'  or model_name == 'models/model_91_d80p':
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
        if model_name == 'models/model_91_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_91_d80p':
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
    elif model_name == 'models/model_102' or model_name == 'models/model_102_d50p' or model_name == 'models/model_102_d80p':
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
        elif model_name == 'models/model_102_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))
    elif model_name == 'models/model_103' or model_name == 'models/model_103_d50p' or model_name == 'models/model_103_d80p':
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
        if model_name == 'models/model_103_d50p':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_103_d80p':
            model.add(Dropout(0.8))
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
    elif model_name == 'models/model_105' or model_name == 'models/model_105_d50p' or model_name == 'models/model_105_d80p':
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
    elif model_name == 'models/vgg16' or model_name == 'vgg16_d50p' or model_name == 'vgg16_d80p':
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
    elif model_name == 'models/vgg19' or model_name == 'vgg19_d50p' or model_name == 'vgg19_d80p':
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

    print(model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(),
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
        model.compile(optimizer=keras.optimizers.Adam(),
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
    with open(train_data, 'rb') as fin:
        x_train, y_train = pickle.load(fin)
    with open(val_data, 'rb') as fin:
        x_val, y_val = pickle.load(fin)

#    x_train = x_train / norm_const
 #   x_val = x_val / norm_const
    model.fit(x_train, y_train, batch_size=128, epochs=100,
              validation_data= (x_val, y_val),
              callbacks=[history,
                         keras.callbacks.EarlyStopping(monitor='acc', patience=4)])

    # Saving history into file
    model.save(model_name + '.h5py')
    with open(model_name + '.aux_data', 'wb') as fout:
        pickle.dump((history.test_losses,history.train_losses,history.test_acc,history.train_acc), fout)

# time taken to run model
end = time.time()
print(end - start)

y_test = []
x_test = []
predict = False
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