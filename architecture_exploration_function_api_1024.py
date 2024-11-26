# importing libraries
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model
import pandas as pd
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

model_name = 'models/model_91_d80p'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# variables
reset_model = False
data = 'C:/Users/Manish/projects/tiya/scienceFair-2019/data/data_with_hv_flip_train'
val_data = 'C:/Users/Manish/projects/tiya/scienceFair-2019/data/data_wiht_hv_flip_val'
AdamOptimzer = False
RMSprop = False

# normalization constant
norm_const = np.array([255.0])
norm_const = norm_const.astype('float16')

# getting the train data
with open(data, 'rb') as fin:
    x_train, y_train = pickle.load(fin)
with open(val_data, 'rb') as fin:
    x_val, y_val = pickle.load(fin)

# Running different CNN architectures
if not os.path.isfile(model_name + '.h5py') or reset_model:
    history.resetHistory(True)
    # 91_d80p
    if model_name=='models/model_91_d80p':
        input_ = Input(shape=(1024, 1024, 1))
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(input_)
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(model)
        model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
        model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(model)
        model = Dropout(0.8)(model)
        model = Flatten()(model)
        out = Dense(1, activation=tf.nn.sigmoid)(model)
        model = Model(input_, outputs=out)

    # RMSprop
    if RMSprop:
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    # SGD Optimizer
    else:
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# if model already exists retrain the model from where it left off
else:
    # history class defined above
    history.resetHistory(False)

    print("Loading model from %s" %(model_name + '.h5py'))
    model = load_model(model_name + '.h5py')
    print (model.summary())
    # Adam Optimizer
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr=0.000005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    # RMS prop
    elif RMSprop:
        model.compile(optimizer=keras.optimizers.RMSprop(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    # SGD Optimizer
    else:
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # Load all previous data
    with open(model_name + '.aux_data', 'rb') as fin:
        test_losses, train_losses, test_acc, train_acc = pickle.load(fin)
    history.test_losses = test_losses
    history.train_losses = train_losses
    history.test_acc = test_acc
    history.train_acc = train_acc
    print (len(history.test_losses))

# plot the model
plot_model(model, to_file=model_name + '.png')

x_train = x_train / norm_const
x_val = x_val / norm_const

start = time.time()
print(x_train.dtype)
# training the model
for i in range(5):
    model.fit(x_train, y_train, batch_size=8, epochs=1, validation_data=(x_val, y_val),
              callbacks=[history,
                         keras.callbacks.EarlyStopping(monitor='acc', patience=4)])

    # Saving history into file
    model.save(model_name + '_' + str(i) + '.h5py')
    with open(model_name + '_' + str(i) + '.aux_data', 'wb') as fout:
        pickle.dump((history.test_losses,history.train_losses,history.test_acc,history.train_acc), fout)

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