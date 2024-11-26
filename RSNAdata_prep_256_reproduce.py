import pandas as pd
import pydicom
import os
import pickle
from skimage.transform import resize
import sys
from sklearn.model_selection import train_test_split

data_train = "data/rsna_train_data_with_256.dat"
data_test = "data/rsna_test_data_with_256.dat"
forceRun = True

if not os.path.isfile(data_train) or forceRun:
    tr = pd.read_csv('data/stage_2_train_labels.csv')
    tr['aspect_ratio'] = (tr['width'] / tr['height'])
    tr['area'] = tr['width'] * tr['height']

    def get_info(patientId, tr, root_dir='data/stage_2_train_images/'):
        fn = os.path.join(root_dir, f'{patientId}.dcm')
        dcm_data = pydicom.read_file(fn)
        a = tr.loc[lambda tr: tr.patientId==patientId, :]
        boxes = []
        for i in range(len(a)):
            boxes.append((a.iloc[i]['x'],a.iloc[i]['y'],a.iloc[i]['width'],a.iloc[i]['height']))
        dcm_pixels = resize(dcm_data.pixel_array, (256, 256))
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

    # train, test split
    train, test = train_test_split(tf, train_size=0.8, test_size=0.2, random_state=1000000000)

    # saving train into data file
    with open(data_train, "wb") as fout:
        print("Saving data file into %s" % (data_train))
        pickle.dump((train), fout)

    # saving test into data file
    with open(data_test, "wb") as fout:
        print("Saving data file into %s" % (data_test))
        pickle.dump((test), fout)
else:
    # getting train set from data file
    print("Reading data from file %s" % (data_train))
    with open(data_train, "rb") as fin:
        train = pickle.load(fin)

    # getting test set from data file
    print("Reading data from file %s" % (data_test))
    with open(data_test, "rb") as fin:
        test = pickle.load(fin)