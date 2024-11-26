import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import keras
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
from pympler.tracker import SummaryTracker

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

tracker = SummaryTracker()
history = LossHistory()
model_name = 'models/model_101'
#model_name = 'vgg19_d80p'
#model_name = 'models/model_6_d50p'
# 40 epocs: acc: 0.8280 - val_loss: 0.4128 - val_acc: 0.8101
#model_name = 'models/model_7'
# 40 epocs: acc: 0.8232 - val_loss: 0.4118 - val_acc: 0.8129
#model_name = 'models/model_8'
#loss: 0.3842 - acc: 0.8277 - val_loss: 0.4142 - val_acc: 0.8084
#model_name = 'models/model_11'
#loss: 0.3456 - acc: 0.8529 - val_loss: 0.4745 - val_acc: 0.7977
#model_name = 'models/model_11_d50p'
#acc: 0.8382 - val_loss: 0.4239 - val_acc: 0.8033

dataAugumentation = True
num_images_per_iteration = 1024
reset_model = False
plot_only = False
data = 'data/data_with_hv'
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
    elif model_name == 'models/model_91' or model_name == 'models/model_91_d50p' or model_name == 'models/model_91_d80p':
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

    elif model_name == 'models/model_102' or model_name == 'models/model_102_d50p' or model_name == 'models/model_102_d80p_aug':
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

    elif model_name == 'vgg16' or model_name == 'vgg16_d50p' or model_name == 'vgg16_d80p':
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
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))

    print(model.summary())
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


else:
    history.resetHistory(False)
    print("Loading model from %s" %(model_name + '.h5py'))
    model = load_model(model_name + '.h5py')
    print (model.summary())
    model.compile(optimizer=keras.optimizers.SGD(decay=0.5),
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

start = time.time()
if not plot_only:
    if not dataAugumentation:
        with open(data + '_0', 'rb') as fin:
            x_train, y_train = pickle.load(fin)

        with open(data + '_1', 'rb') as fin:
            x_t1, y_t1 = pickle.load(fin)

        x_train = np.concatenate((x_train, x_t1))
        y_train = np.concatenate((y_train, y_t1))
        del x_t1, y_t1
        norm_const = np.array([255])
        norm_const.astype('float16')
        x_train = x_train/norm_const

    if dataAugumentation:
        x_all = []
        y_all = []
        for i in range(4):
            with open(data + '_aug_rot20_' + str(i), 'rb') as fin:
                x_train, y_train = pickle.load(fin)
            if i == 0:
                x_all = x_train
                y_all = y_train
            else:
                x_all = np.concatenate((x_all,x_train))
                y_all = np.concatenate((y_all,y_train))
        # re-shuffle
        #choices = np.arange(y_train.shape[0])
        #np.random.shuffle(choices)
        #x_all = x_train[choices]
        #y_all = y_train[choices]
        # train/validation split
        #aaa = int(y_train.shape[0]*0.8)
        #x_train, x_test = x_train[:aaa, :], x_train[aaa:, :]
        #y_train, y_test = y_train[:aaa, :], y_train[aaa:, :]
        #print (x_train.shape)
        #print (y_train.shape)
        #print (x_test.shape)
        #print (y_test.shape)
        x_train = x_all
        y_train = y_all
        del x_all, y_all
        batch_size = 512
        datagen = ImageDataGenerator(rotation_range=20)
        datagen.fit(x_train,augment=True)
        model.fit_generator(datagen, steps_per_epoch=y_train.shape[0]/batch_size, epochs=40, verbose=2)

        #num_images = y_train.shape[0]
        #num_itr_per_epoch = int(np.ceil(num_images/num_images_per_iteration))
        #print ("Number of iterations per epoch is %d" %(num_itr_per_epoch))
        #for e in range(40):
            # print ('Epoch', e)
            # for itr in range(num_itr_per_epoch):
            #     print("Iteration Number: " + str(itr))
            #     batches = 0
            #     if itr == num_itr_per_epoch-1:
            #         xx_train = x_train[itr*num_images_per_iteration:]
            #         yy_train = y_train[itr*num_images_per_iteration:]
            #     else:
            #         xx_train = x_train[itr*num_images_per_iteration:(itr+1)*num_images_per_iteration+1]
            #         yy_train = y_train[itr*num_images_per_iteration:(itr+1)*num_images_per_iteration+1]
            #     print(xx_train.shape)
            #     datagen.fit(xx_train, augment=True)
            #     for xx_train, yy_train in datagen.flow(xx_train, yy_train, batch_size=num_images_per_iteration):
            #         xx_train = xx_train.astype('float16')
            #         model.fit(xx_train, yy_train, batch_size=32, epochs=1,
            #                 validation_data=(x_test,y_test),
            #                 callbacks=[history,
            #                     keras.callbacks.EarlyStopping(monitor='acc', patience=4)])
            #         tracker.print_diff()
            #         break

    else:
       # x_train = np.reshape(x_train, (x_train.shape[0], 256, 256, 1))
        print(y_train.sum())
        print(x_train.shape)
        print(y_train.shape)
        model.fit(x_train, y_train, batch_size=8, epochs=40,
                  validation_split=0.2,
                  callbacks=[history,
                             keras.callbacks.EarlyStopping(monitor='acc', patience=8)])

    # Saving history into file
    model.save(model_name + '.h5py')
    with open(model_name + '.aux_data', 'wb') as fout:
        pickle.dump((history.test_losses,history.train_losses,history.test_acc,history.train_acc), fout)

end = time.time()
print(end - start)

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