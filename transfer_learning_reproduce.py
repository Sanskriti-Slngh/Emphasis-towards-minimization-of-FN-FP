import os

import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model
import pandas as pd
import sys
import random
import time
from sklearn.metrics import confusion_matrix
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
model_name = 'models/model_transfer_learning_1'
frozen_model = 'models/model_multi_class'

reset_model = False
plot_only = True
train_data = 'data/train_data_with_256'
val_data = 'data/val_data_with_256'
AdamOptimzer = True
predict = False

if not os.path.isfile(model_name + '.h5py') or reset_model:
    history.resetHistory(True)
    model = keras.models.Sequential()
    freezemodel = load_model(frozen_model + '.h5py')
    print(freezemodel.summary())
    if model_name == 'models/model_transfer_learning_1':
        model.add(freezemodel)
        model.add(Dense(1, activation=tf.nn.sigmoid, name="dense_2"))
    elif model_name == 'models/model_transfer_learning_2':
        model.add(freezemodel)
        model.add(Dense(2, activation=tf.nn.sigmoid, name="dense_2"))
        model.add(Dense(1, activation=tf.nn.sigmoid, name="dense_3"))
    elif model_name == 'models/model_transfer_learning_3':
        model.add(freezemodel)
        model.add(Dense(4, activation=tf.nn.sigmoid, name="dense_2"))
        model.add(Dense(1, activation=tf.nn.sigmoid, name="dense_3"))
    else:
        print ("Model not defined")
        exit()

    print(model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr= 0.000001, decay=0.1),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=keras.optimizers.SGD(lr=0.00001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

else:
    history.resetHistory(False)
    print("Loading model from %s" %(model_name + '.h5py'))
    model = load_model(model_name + '.h5py')
    print (model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr= 0.000001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
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

# normalization constant
norm_const = np.array([255.0])
norm_const = norm_const.astype('float16')

start = time.time()
if not plot_only:
    if not predict:
        # getting the train data
        with open(train_data, 'rb') as fin:
            x_train, y_train = pickle.load(fin)
        with open(val_data, 'rb') as fin:
            x_val, y_val = pickle.load(fin)

        x_all = x_train
        y_all = y_train
        x_v = x_val
        y_v = y_val
        del x_train,y_train, x_val, y_val

        # shuffle again
        indices = np.arange(x_all.shape[0])
        np.random.shuffle(indices)
        x_train = x_all[indices]
        y_train = y_all[indices]
        indices = np.arange(x_v.shape[0])
        np.random.shuffle(indices)
        x_val = x_v[indices]
        y_val = y_v[indices]
        del x_all, y_all, x_v, y_v

        print ("Shapes of x train and y train: ")
        print (x_train.shape)
        print (y_train.shape)
        print (y_train.sum())
        print(y_train)
        print (x_train.dtype)
        print (y_train.dtype)
        print(y_train.shape)
        model.fit(x_train, y_train, batch_size=128, epochs=10,
                    validation_data=(x_val, y_val),
                    callbacks=[history,
                               keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)])

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