import pandas as pd
import pydicom
import os
import pickle
import cv2
import random
import sys
from sklearn.model_selection import train_test_split

# fix the seed so that we get same train/test splitpy
random.seed(10000)

data = "D:/ScienceFairProjects/2018-19/rawdata_256_with_hv_flips_for_1"
forceRun = True

# split train test
train, test = train_test_split(train_size=0.8, test_size=0.2)

if not os.path.isfile(data) or forceRun:
    tr = pd.read_csv('D:/ScienceFairProjects/2018-19/data/stage_2_train_labels.csv')
    tr['aspect_ratio'] = (tr['width'] / tr['height'])
    tr['area'] = tr['width'] * tr['height']

    def get_info(patientId, tr, root_dir='D:/ScienceFairProjects/2018-19/data/stage_2_train_images/'):
        fn = os.path.join(root_dir, f'{patientId}.dcm')
        dcm_data = pydicom.read_file(fn)
        a = tr.loc[lambda tr: tr.patientId==patientId, :]
        boxes = []
        for i in range(len(a)):
            boxes.append((a.iloc[i]['x'],a.iloc[i]['y'],a.iloc[i]['width'],a.iloc[i]['height']))
        dcm_pixels = cv2.resize(dcm_data.pixel_array, (256, 256))
        #print(dcm_pixels.shape)
        #print(dcm_pixels.size)
        return {'age': int(dcm_data.PatientAge),
                'gender': dcm_data.PatientSex,
                'id': os.path.basename(fn).split('.')[0],
                'pixel_spacing': float(dcm_data.PixelSpacing[0]),
                'boxes':boxes,
                'Modality':dcm_data.Modality,
                'pixel_data': dcm_pixels}


    patient_ids = list(tr.patientId.unique())
    result = []
    for i in patient_ids:
        result.append(get_info(i, tr))

    demo = pd.DataFrame(result)
    demo['gender'] = demo['gender'].astype('category')
    demo['age'] = demo['age'].astype(int)
    tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left').drop(columns=['id','x','y','width','height']))
    tf = tr.drop_duplicates(subset=['patientId'])
    print (sys.getsizeof(tf))
    # train and test split
    # 80-20 split
    with open(data, "wb") as fout:
        print ("Saving data file into %s" %(data))
        pickle.dump((tf),fout)
else:
    print("Reading data from file %s" % (data))
    with open(data, "rb") as fin:
        tf = pickle.load(fin)



