import pandas as pd
import pydicom
import os
import pickle
import random
import sys
from PIL import Image
import cv2
import numpy as np

# name of dataset contianing all images
d = 'D:/xraydata/nih/images/'

path, dirs, files = next(os.walk(d))
print (len(files))
#exit()

data = pd.read_csv('D:/xraydata/nih/Data_Entry_2017.csv')
#data['target'] = data['Finding Labels'].str.contains('(Pneumothorax|Pneumonia)', regex=True)
aaa =  (list(data['Finding Labels'].unique()))
diseases = []
for a in aaa:
    for x in a.split('|'):
        diseases.append(x)
diseases = set(diseases)
for x in diseases:
    print (x)
print (len(diseases))
exit()
#print (data['target'].sum())
#print (data.info())
a = 0
count = 0
for disease in diseases:
    print(disease)
    X = []
    data['target'] = data['Finding Labels'].str.contains(disease, regex=True)
    #print (data['target'])
    for x in (data.loc[data['target']]['Image Index']):
        if not os.path.isfile(d+x):
            a = a+1
            continue
        count = count + 1
        if count%20 == 0:
            if count%4000 == 0:
                print('\n')
            print('.', end='', flush=True)

        img = cv2.imread(d+x)
        img = cv2.resize(img, (256,256))
        img = img[:,:,0]
        X.append(img)
    print('\n'+"Number of skipped images: " + str(a))
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], 256, 256, 1))
    with open('D:/xraydata/nih/' + str(disease) + '.data', 'wb') as fin:
        pickle.dump(X, fin)
    print (X.shape)
    print (sys.getsizeof(X))
