import pickle
import numpy as np
from keras.models import load_model
from statistics import mode
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
import random

# loading all models for ensembling
model_1 = load_model('Ensembling_models/0%/model_91_d80p.h5py')
model_2 = load_model('Ensembling_models/0%/model_102_d80p.h5py')
model_3 = load_model('Ensembling_models/0%/vgg16.h5py')
model_4 = load_model('Ensembling_models/0%/model_transfer_learning_5.h5py')
model_5 = load_model('Ensembling_models/0%/model_103_d80p.h5py')

# Variables
Max_Voting = True
Averaging = False
weighted_averaging = False
y_test = []
x_test = []
norm_const = np.array([255.0])
norm_const = norm_const.astype('float16')

# Getting the Test Set
with open('data/rawdata_exp_256_test.dat', 'rb') as fin:
    test = pickle.load(fin)
# Preparing the test set
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
x_test_show = x_test
x_test = x_test/norm_const

# Predicitng on all Ensembling Models
pred1 = model_1.predict(x_test)
pred2 = model_2.predict(x_test)
pred3 = model_3.predict(x_test)
pred4 = model_4.predict(x_test)
pred5 = model_5.predict(x_test)
# Getting the y_hat
y_hat = np.array([])
for i in range(0, len(test)):
    y_hat = np.append(y_hat, max([pred1[i,0], pred2[i,0], pred3[i,0], pred4[i,0], pred5[i,0]]))

# Ensembling
final_pred = np.array([])
if Max_Voting:
    pred1 = pred1 > 0.5
    pred2 = pred2 > 0.5
    pred3 = pred3 > 0.5
    pred4 = pred4 > 0.5
    pred5 = pred5 > 0.5
    for i in range(0,len(test)):
        final_pred = np.append(final_pred, mode([pred1[i,0], pred2[i,0], pred3[i,0], pred4[i,0], pred5[i,0]]))
    print(final_pred.shape)
if Averaging:
    final_pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5
    print (final_pred)
    final_pred = final_pred >0.5
    print (final_pred)
if weighted_averaging:
    final_pred = (pred1 * 0.2 + pred2 * 0.6 + pred3 * 0.3 + pred4 * 0.5 + pred5 * 0.5)
    final_pred = final_pred >0.5

# fpr - false postive rate tpr - true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, final_pred, pos_label=0)

# Print ROC curve
plt.plot(fpr,tpr)

# Print AUC
auc = metrics.auc(tpr,fpr)
print('AUC:', auc)

# Getting the confusion matrix
cm = confusion_matrix(y_test, final_pred)
print(cm)
df_cm = pd.DataFrame(cm, index= ['True 0','True 1'], columns= ['Predicted 0','Predicted 1'])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt= 'g')
acc = (cm[0,0] + cm[1,1])/(cm[1,0] + cm[0,1] + cm[0,0] + cm[1,1])
print("Accuracy:   " + str(acc))
print(cm[1,0])
rec = cm[1,1]/(cm[1,1] + cm[1,0])
pre = cm[1,1]/(cm[1,1] + cm[0,1])
f1 = 2 * ((pre*rec)/(pre+rec))

# results
print('Precision:    ' + str(pre))
print('Recall:    ' + str(rec))
print('F1 score:    ' + str(f1))
plt.show()

# displaying images
final_pred = np.array(final_pred)
for k in range(16):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in range(2):
        for j in range(2):
            index = random.randint(0, len(y_test))
            img = x_test_show[index, :, :, 0]
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].axis('off')
            ax[i][j].text(0, 0, "P=" + str(final_pred[index]) + '   R=' + str(y_test[index]))
    plt.show()

# Graphs
p = plt.hist(y_hat, bins=20)
for i in range(20):
    plt.text(p[1][i],p[0][i],str(p[0][i]))

plt.show()