# importing libraries
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import datetime
from sklearn.model_selection import train_test_split

# getting and saving the data
data_in = "data/rawdata_exp_1024_train.dat"
data_out = 'C:/Users/Manish/projects/tiya/scienceFair-2019/data/data_with_hv_flip_train'
data_out_val = 'data/data_wiht_hv_flip_val'

x_train = []
y_train = []
aug_train = []

# 0 - No, 1 - h flip, 2 - v flip, 3 - hv flip
with open(data_in, 'rb') as fin:
    set = pickle.load(fin)
    number = set['patientId'].count()
    for i in range(number):
        y_train.append([set.iloc[i]['Target']])
        x_train.append([set.iloc[i]['pixel_data']])
        aug_train.append(0)
        if set.iloc[i]['Target']:
            y_train.append([1])
            y_train.append([1])
            img = set.iloc[i]['pixel_data']
            # horizontal flip
            img_h = np.flip(img, axis = 1)
            # horizontal flip
            img_v = np.flip(img, axis = 0)
            # horizontal and vertical flip
            num = random.randint(0, 1)
            if num == 1:
                img_hv = np.flip(img_h, axis=0)
                x_train.append([img_hv])
                y_train.append([1])
                aug_train.append(3)
            x_train.append([img_h])
            aug_train.append(1)
            x_train.append([img_v])
            aug_train.append(2)

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], 1024, 1024, 1))
y_train = np.array(y_train)

# plotting images, horizontal, vertical
if True:
    for k in range(4):
        fig,ax = plt.subplots(nrows=2,ncols=2)
        for i in range(2):
            for j in range(2):
                index = random.randint(0,y_train.shape[0])
                img = x_train[index,:,:,0]
                if aug_train[index] == 1 or aug_train[index] == 3:
                    img = np.flip(img,axis=1)
                if aug_train[index] == 2 or aug_train[index] == 3:
                    img = np.flip(img, axis=0)

                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].axis('off')
                ax[i][j].text(0,0,str([y_train[index][0]]) + '_' + str(aug_train[index]))

        plt.show()

#splitting into train and validation
x_val_data = []
y_val_data = []
x_train_data = []
y_train_data = []
for i in range(0,y_train.shape[0]):
    random_number = random.randint(1,101)
    img = x_train[i, :, :, 0]
    y_value = y_train[i]
    if random_number > 80:
        x_val_data.append(img)
        y_val_data.append(y_value)
    else:
        x_train_data.append(img)
        y_train_data.append(y_value)

x_val_data = np.array(x_val_data)
x_val_data = np.reshape(x_val_data, (x_val_data.shape[0], 1024, 1024, 1))
y_val_data = np.array(y_val_data)
x_train_data = np.array(x_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], 1024, 1024, 1))
y_train_data = np.array(y_train_data)

print (datetime.time())
print (sys.getsizeof(x_train))

# saving new augumented data into file
with open(data_out, 'wb') as fout:
    pickle.dump((x_train_data, y_train_data), fout, protocol=4)

with open(data_out_val, 'wb') as fout:
    pickle.dump((x_val_data, y_val_data), fout, protocol= 4)

print (datetime.time())

print(x_val_data.shape)
print(y_val_data.shape)
print(x_train_data.shape)
print(y_train_data.shape)

if True:
    for k in range(4):
        fig,ax = plt.subplots(nrows=2,ncols=2)
        for i in range(2):
            for j in range(2):
                index = random.randint(0,y_train_data.shape[0])
                img = x_train[index,:,:,0]
              #  if aug_train[index] == 1 or aug_train[index] == 3:
               #     img = np.flip(img,axis=1)
                #if aug_train[index] == 2 or aug_train[index] == 3:
                 #   img = np.flip(img, axis=0)

                ax[i][j].imshow(img, cmap='gray')
                ax[i][j].axis('off')
                ax[i][j].text(0,0,str([y_train[index][0]]) + '_' + str(aug_train[index]))

        plt.show()
