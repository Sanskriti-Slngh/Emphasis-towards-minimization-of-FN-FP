import pickle
import numpy as np
from keras.backend import binary_crossentropy, mean
from keras.models import load_model
import keras
from statistics import mode
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random

## fixing randomization for repeat
random.seed(100213)

# user variables
continue_this_model = True

# History
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

data = 'C:/Users/Manish/projects/tiya/scienceFair-2019/data/data_with_hv_flip_train'
val_data = 'C:/Users/Manish/projects/tiya/scienceFair-2019/data/data_wiht_hv_flip_val'
z = 'model_104_d50p'
#model_name = 'Saved_previous_models/' + z
model_name = 'models/' + z
model = load_model(model_name +'.h5py')
q = 'models/false_negatives/100%/' + z

print (model.summary())

norm_const = np.array([255.0])
norm_const = norm_const.astype('float16')

with open(data, 'rb') as fin:
    x_train, y_train = pickle.load(fin)
with open(val_data, 'rb') as fin:
    x_val, y_val = pickle.load(fin)

print(x_train.shape)
x_train = x_train/norm_const
print('here')
y_hat = model.predict(x_train, batch_size=16)
pred1 = y_hat > 0.5
print('finished predicting')
cm = confusion_matrix(y_train, pred1)
print(cm)
print(pred1.shape)
for i in range(y_train.shape[0]):
    if pred1[i] == 1 and y_train[i] == 0:
        ran_num = random.randint(1,100)
        if ran_num <= 100:
            y_train[i] = 1
        else:
            continue

cm = confusion_matrix(y_train, pred1)
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

if continue_this_model:
    model = load_model(q + '.h5py')

history.resetHistory(False)
# lr = 0.1 - too high
# lr = 0.01
# lr = 0.001
# lr = 0.0001
# lr = 0.00001
model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

with open(model_name + '.aux_data', 'rb') as fin:
    test_losses, train_losses, test_acc, train_acc = pickle.load(fin)
history.test_losses = test_losses
history.train_losses = train_losses
history.test_acc = test_acc
history.train_acc = train_acc
print (len(history.test_losses))
model.fit(x_train, y_train, batch_size=8, epochs=9, validation_data=(x_val, y_val),
          callbacks=[history,
                     keras.callbacks.EarlyStopping(monitor='acc', patience=4)])

# Saving history into file
model.save(q + '.h5py')
with open(q + '.aux_data', 'wb') as fout:
    pickle.dump((history.test_losses, history.train_losses, history.test_acc, history.train_acc), fout)

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