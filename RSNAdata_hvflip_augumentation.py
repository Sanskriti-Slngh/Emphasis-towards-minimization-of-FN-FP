import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import datetime
from keras.preprocessing.image import ImageDataGenerator

data_in = "data/rawdata_exp_256"
data_out = 'data/data_with_hv'

x_train = []
y_train = []
aug_train = []
# 0 - No, 1 - h flip, 2 - v flip, 3 - hv flip
for i in range(4):
    with open(data_in + '_train_set_' + str(i) + '.dat', 'rb') as fin:
        set = pickle.load(fin)
        # Train set 0
        number = set['patientId'].count()
        #        x_train = []
        #        y_train = []
        for i in range(number):
            y_train.append([set.iloc[i]['Target']])
            x_train.append([set.iloc[i]['pixel_data']])
            aug_train.append(0)
            if set.iloc[i]['Target']:
                y_train.append([1])
                y_train.append([1])
                # h flip
                img = set.iloc[i]['pixel_data']
                img_h = np.flip(img, 1)
                img_v = np.flip(img, 0)
                num = random.randint(0, 1)
                if num == 1:
                    img_hv = np.flip(img_h, 0)
                    x_train.append([img_hv])
                    y_train.append([1])
                    aug_train.append(3)
                x_train.append([img_h])
                aug_train.append(1)
                x_train.append([img_v])
                aug_train.append(2)

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], 256, 256, 1))
y_train = np.array(y_train)

if True:
    for k in range(4):
        fig,ax = plt.subplots(nrows=2,ncols=2)
        for i in range(2):
            for j in range(2):
                index = random.randint(0,y_train.shape[0])
                img = x_train[index,:,:,0]
                if aug_train[index] == 1 or aug_train[index] == 3:
                    img = np.flip(img,1)
                if aug_train[index] == 2 or aug_train[index] == 3:
                    img = np.flip(img, 0)

                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].axis('off')
                ax[i][j].text(0,0,str([y_train[index][0]]) + '_' + str(aug_train[index]))

        plt.show()

print (datetime.time())
print (sys.getsizeof(x_train))

with open(data_out, 'wb') as fout:
    pickle.dump((x_train, y_train), fout)

print (datetime.time())