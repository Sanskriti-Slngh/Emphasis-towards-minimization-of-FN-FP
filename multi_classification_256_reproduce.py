import os

import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import plot_model
import pandas as pd
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
model_name ='models/model_multi_class'

reset_model = False
plot_only = True
train_data = 'data/train_data_with_256'
val_data = 'data/val_data_with_256'
AdamOptimzer = False
RMSprop = False
predict = False
num_of_folds = 10
plot_results = False
include_nih_data = True
nihd = 'F:/xraydata/nih/'
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
    if model_name == 'models/model_105' or model_name == 'models/model_multi_class' or model_name == 'models/model_105_d80p_multi_classification' or model_name == 'models/model_105_adam' or model_name == 'models/model_105_rms_prop' or model_name == 'models/model_105_k_fold':
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
        if model_name == 'models/model_multi_class':
            model.add(Dropout(0.5))
        elif model_name == 'models/model_105_d80p':
            model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(15, activation=tf.nn.sigmoid))

    else:
        print ("Model not defined")
        exit()

    print(model.summary())
    if AdamOptimzer:
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0000001),
                      loss='categorical_crossentropy',
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
        model.compile(optimizer=keras.optimizers.Adam(lr=0.000001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    elif RMSprop:
        model.compile(optimizer=keras.optimizers.RMSprop(),
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
        if include_nih_data:
            x_1 = []
            y_1 = []
            for i,disease in enumerate(diseases):
                with open(nihd+disease+".data", 'rb') as fin:
                    x_tmp = pickle.load(fin)
                    if len(x_1) == 0:
                        x_1 = x_tmp
                        y_1 = np.zeros((x_tmp.shape[0], 1))
                        y_1.fill(d2idex[disease])
                    else:
                        x_1 = np.concatenate((x_1,x_tmp))
                        y_tmp = np.zeros((x_tmp.shape[0], 1))
                        y_tmp.fill(d2idex[disease])
                        y_1 = np.concatenate((y_1,y_tmp))

            print (x_1.shape)
            print (y_1.shape)

            indices = np.arange(x_1.shape[0])
            np.random.shuffle(indices)

            print ("Prior to adding nih images")
            print(x_1.shape)
            print(y_1.shape)

            x_all = x_1
            y_all = y_1
            del x_1,y_1

            # shuffle again
            indices = np.arange(x_all.shape[0])
            np.random.shuffle(indices)
            x_train = x_all[indices]
            y_train = y_all[indices]
            del x_all, y_all

            print ("After adding nih images")
            print (x_train.shape)
            print (y_train.shape)
            x_train = x_train/norm_const
            y_train = keras.utils.to_categorical(y_train, num_classes=15, dtype='float16')
            print(y_train)
            print (x_train.dtype)
            print (y_train.dtype)
            print(y_train.shape)
            model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2,
                      callbacks=[history,
                                 keras.callbacks.EarlyStopping(monitor='acc', patience=8)])

            # Saving history into file
            model.save(model_name + '.h5py')
            with open(model_name + '.aux_data', 'wb') as fout:
                pickle.dump((history.test_losses, history.train_losses, history.test_acc, history.train_acc), fout)

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
        print('Precision:    ' + str(pre))
        print('Recall:    ' + str(rec))
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