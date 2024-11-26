import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import keras
from keras.utils import to_categorical
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
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
model_name = 'transfer_learning_models/model_5'
#model_name = 'models/model_103_d50p_multi_class'
frozen_model = 'models/model_105_d80p_multi_classification_0'

dataAugumentation = False
reset_model = False
plot_only = False
data = 'data/data_with_hv'
AdamOptimzer = True
predict = True
num_of_folds = 10
plot_results = False
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

if not os.path.isfile(model_name + '.h5py') or reset_model:
    history.resetHistory(True)
    model = keras.models.Sequential()
    freezemodel = keras.models.Sequential()
    freezemodel = load_model(frozen_model + '.h5py')
    print(freezemodel.summary())
    #for layer in freezemodel.layer[:]:
     #   layer.trainable = False
    #freezemodel.trainable = False
    if model_name == 'models/model_transfer_learning_1':
        # Max Epocs = 9 - loss: 0.6855 - acc: 0.4921 - val_loss: 0.6861 - val_acc: 0.4914
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

    elif model_name == 'models/model_transfer_learning_4':
        model.add(freezemodel)
        # model.add(Dense(2048, activation=tf.nn.sigmoid, name="dense_2"))
        # model.add(Dense(2048, activation=tf.nn.sigmoid, name="dense_3"))
        # model.add(Dense(1024, activation=tf.nn.sigmoid, name="dense_4"))
        # model.add(Dense(1024, activation=tf.nn.sigmoid, name="dense_5"))
        # model.add(Dense(512, activation=tf.nn.sigmoid, name="dense_6"))
        # model.add(Dense(512, activation=tf.nn.sigmoid, name="dense_7"))
        # model.add(Dense(256, activation=tf.nn.sigmoid, name="dense_8"))
        # model.add(Dense(256, activation=tf.nn.sigmoid, name="dense_9"))
        # model.add(Dense(128, activation=tf.nn.sigmoid, name="dense_10"))
        # model.add(Dense(128, activation=tf.nn.sigmoid, name="dense_11"))
        model.add(Dense(64, activation=tf.nn.sigmoid, name="dense_12"))
        model.add(Dense(64, activation=tf.nn.sigmoid, name="dense_13"))
        model.add(Dense(1, activation=tf.nn.sigmoid, name="dense_14"))

    elif model_name == 'models/model_transfer_learning_5':
        model.add(freezemodel)
        model.add(Dense(1, activation=tf.nn.sigmoid, name="dense_14"))
    else:
        print ("Model not defined")
        exit()

    print(model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr= 0.000001, decay=0.1),
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
        model.compile(optimizer=keras.optimizers.Adam(lr= 0.0000001, decay=0.1),
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
        with open(data, 'rb') as fin:
            x_train, y_train = pickle.load(fin)

        x_all = x_train
        y_all = y_train
        del x_train,y_train

        # shuffle again
        indices = np.arange(x_all.shape[0])
        np.random.shuffle(indices)
        x_train = x_all[indices]
        y_train = y_all[indices]
        del x_all, y_all

        print ("Shapes of x train and y train: ")
        print (x_train.shape)
        print (y_train.shape)
        print (y_train.sum())
        if False:
            for k in range(4):
                fig, ax = plt.subplots(nrows=2, ncols=2)
                for i in range(2):
                    for j in range(2):
                        if i%2==0:
                            index = random.randint(0, x_train.shape[0])
                            img = x_train[index, :, :, 0]
                            disease = diseases[int(y_train[index,0])]
                        else:
                            index = random.randint(0, x_train.shape[0])
                            img = x_train[index, :, :, 0]
                            disease = diseases[int(y_train[index, 0])]
                        ax[i][j].imshow(img, cmap='gray')
                        ax[i][j].axis('off')
                        ax[i][j].text(0, 0, disease)

                plt.show()
        x_train = x_train/norm_const
        print(y_train)
        print (x_train.dtype)
        print (y_train.dtype)
        print(y_train.shape)
        model.fit(x_train, y_train, batch_size=8, epochs=40,
                    validation_split=0.2,
                    callbacks=[history,
                                keras.callbacks.EarlyStopping(monitor='acc', patience=8)])

        # Saving history into file
        model.save(model_name + '.h5py')
        with open(model_name + '.aux_data', 'wb') as fout:
            pickle.dump((history.test_losses,history.train_losses,history.test_acc,history.train_acc), fout)

    y_test = []
    x_test = []
    if predict:
        with open('data/rawdata_exp_256_test.dat', 'rb') as fin:
            test = pickle.load(fin)
        with open(data, 'rb') as fin:
            x_train, y_train = pickle.load(fin)
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
        y_pred = model.predict(x_test)
        print (y_pred)
        y_pred = y_pred > 0.5

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
        pre = cm[1,1]/(cm[1,1] + cm[1,0])
        rec = cm[1,1]/(cm[1,1] + cm[0,1])
        f1 = 2 * ((pre*rec)/(pre+rec))
        print('Precision:    ' + str(rec))
        print('Recall:    ' + str(pre))
        print('F1 score:    ' + str(f1))
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