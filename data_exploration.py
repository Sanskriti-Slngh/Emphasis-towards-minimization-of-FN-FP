import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pydicom
import numpy as np
import warnings
import multiprocessing
import os
import zipfile
import pickle
import csv

if not os.path.isfile("./.dataCache"):
    tr = pd.read_csv('data/stage_2_train_labels.csv')
    tr['aspect_ratio'] = (tr['width'] / tr['height'])
    tr['area'] = tr['width'] * tr['height']

    def get_info(patientId, root_dir='data/stage_2_train_images/'):
        fn = os.path.join(root_dir, f'{patientId}.dcm')
        dcm_data = pydicom.read_file(fn)
        return {'age': int(dcm_data.PatientAge),
                'gender': dcm_data.PatientSex,
                'id': os.path.basename(fn).split('.')[0],
                'pixel_spacing': float(dcm_data.PixelSpacing[0]),
                'mean_black_pixels': np.mean(dcm_data.pixel_array == 0)}

    patient_ids = list(tr.patientId.unique())
    result = []
    for i in patient_ids:
        result.append(get_info(i))
    with open("./.dataCache", "wb") as fout:
        pickle.dump((tr,result),fout)
else:
    with open("./.dataCache", "rb") as fin:
        tr,result = pickle.load(fin)

demo = pd.DataFrame(result)

demo['gender'] = demo['gender'].astype('category')
demo['age'] = demo['age'].astype(int)

tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left').drop(columns='id'))
print(tr.info)

# Function
def line_plot(n, bins, _, x):
    bins1 = []
    length = len(bins) - 1
    for i in range(length):
        a = (bins[i]+bins[i+1])/2
        bins1.append(a)
    x.plot(bins1, n, color = 'orangered')

#Graphs
#Are the classes imbalanced
boxes = tr.groupby('patientId')['Target'].sum()
c = (boxes > 0).value_counts()
b = c.plot.bar()
title = b.set_title('Are the classes imblanced?')
x_axis = b.set_xlabel('Has Pneumonia')
y_axis = b.set_ylabel('Count')
plt.show()
print('Are Classes imbalanced:')
print(c)
print('\n')

# How many cases are there per image
c = boxes.value_counts()
b = c.plot.bar()
title = b.set_title('How many cases are there per image')
x_axis = b.set_xlabel('Number of Boxes in CXR')
y_axis = b.set_ylabel('Count')
plt.show()
print('How many cases are there per image:')
print(c)
print('\n')

# What is the age distribution by gender and target,0?
tf = tr.drop_duplicates(subset=['patientId'])
groups = tf.groupby('Target')
group0 = groups.get_group(0)
group1 = groups.get_group(1)
n_bins = 30
fig, axs = plt.subplots(2, 2)
print(axs.shape)

a = group0.groupby('gender')
a1 = a.get_group('F')
a2 = a.get_group('M')
b = group1.groupby('gender')
b1 = b.get_group('F')
b2 = b.get_group('M')

plt.figtext(.5,.9,'Age VS. Gender', fontsize=18, ha='center')
n, bins, _ = axs[0][0].hist(a1['age'], bins=n_bins, color= 'c')
axs[0][0].set_title('Female, Target 0')
axs[0][0].set_xlabel("Age")
axs[0][0].set_ylabel("Count")
line_plot(n, bins, _,axs[0][0])
n, bins, _ = axs[0][1].hist(a2['age'], bins=n_bins, color= 'b')
axs[0][1].set_title('Male, Target 0')
axs[0][1].set_xlabel("Age")
axs[0][1].set_ylabel("Count")
line_plot(n, bins, _, axs[0][1])
n, bins, _ = axs[1][0].hist(b1['age'], bins=n_bins, color='c')
axs[1][0].set_title('Female, Target 1')
axs[1][0].set_xlabel("Age")
axs[1][0].set_ylabel("Count")
line_plot(n, bins, _, axs[1][0])
n, bins, _ = axs[1][1].hist(b2['age'], bins=n_bins, color= 'b')
axs[1][1].set_title('Male, Target 1')
axs[1][1].set_xlabel("Age")
axs[1][1].set_ylabel("Count")
line_plot(n, bins, _, axs[1][1])
plt.show()


# What are the areas of the bounding boxes by gender?
groups = tf.groupby('gender')
group0 = groups.get_group('F')
group1 = groups.get_group('M')
a = group0['area']
b = group1['area']
a1 = a.dropna()
b1 = b.dropna()


fig, axs = plt.subplots(1, 2)

plt.figtext(.5,.9,'The Areas of the Bounding Boxes by Gender', fontsize=18, ha='center')

n, bins, _ = axs[0].hist(a1, bins=30, color= 'c')
axs[0].set_title('Female')
axs[0].set_xlabel('Areas')
axs[0].set_ylabel('Count')
line_plot(n, bins, _, axs[0])
n, bins, _ = axs[1].hist(b1, bins=30, color= 'b')
axs[1].set_title('Male')
axs[1].set_xlabel('Areas')
axs[1].set_ylabel('Count')
line_plot(n, bins, _, axs[1])
plt.show()



# Pixel Spacing
b = tr['pixel_spacing'].value_counts()
c = b.plot.bar()
title = c.set_title('Pixel Spacing Distribution')
x_axis = c.set_xlabel('Pixel Spacing')
y_axis = c.set_ylabel('Count')
plt.show()
print('Pixel Spacing: ')
print(b)
print('\n')

# How are the bounding box areas distributed by the number of boxes?
boxes = tr.groupby('patientId')['Target'].sum()
fig, axs = plt.subplots(2, 2)
areas = tr.dropna(subset=['area'])
areas_with_count = areas.merge(pd.DataFrame(boxes).rename(columns={'Target': 'bbox_count'}), on='patientId')
boxes = areas_with_count.groupby('bbox_count')
box1 = boxes.get_group(1)
box2 = boxes.get_group(2)
box3 = boxes.get_group(3)
box4 = boxes.get_group(4)
plt.figtext(.5,.9,'Bounding Box Distribution', fontsize=18, ha='center')

n, bins, _ = axs[0][0].hist(box1['area'], bins=100, color= 'b')
axs[0][0].set_title('Bounding Boxes: 1')
axs[0][0].set_xlabel('Areas')
axs[0][0].set_ylabel('Count')
line_plot(n, bins, _, axs[0][0])
n, bins, _ = axs[0][1].hist(box2['area'], bins=100, color= 'g')
axs[0][1].set_title('Bounding Boxes: 2')
axs[0][1].set_xlabel('Areas')
axs[0][1].set_ylabel('Count')
line_plot(n, bins, _, axs[0][1])
n, bins, _ = axs[1][0].hist(box3['area'], bins=100, color= 'orchid')
axs[1][0].set_title('Bounding Boxes: 3')
axs[1][0].set_xlabel('Areas')
axs[1][0].set_ylabel('Count')
line_plot(n, bins, _, axs[1][0])
n, bins, _ = axs[1][1].hist(box4['area'], bins=100, color= 'y')
axs[1][1].set_title('Bounding Boxes: 4')
axs[1][1].set_xlabel('Areas')
axs[1][1].set_ylabel('Count')
line_plot(n, bins, _, axs[1][1])
plt.show()

# Location of boxes
x = np.array(areas['x'].tolist()) + np.array(areas['width'].tolist())/2
y = np.array(areas['y'].tolist()) + np.array(areas['height'].tolist())/2
plt.title("Where is Pneumonia Located")
plt.xlabel("Horizontal distance from top-left")
plt.ylabel("Vertical distance from top-left")
plt.scatter(x, y, color= 'lightpink')
plt.show()
# what does the distribution aspect ratios look like?
fig, axs = plt.subplots(1, 1)
n, bins, _ = axs.hist(tr['aspect_ratio'].dropna(), bins = 50, color= 'lightskyblue')
axs.set_title('The Distribution of Aspect Ratios')
axs.set_xlabel('Aspect Ratios')
axs.set_ylabel('Count')
line_plot(n, bins, _, axs)
plt.show()



