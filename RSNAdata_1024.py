# importing all libraries
import pandas as pd
import pydicom
import os
import pickle
import sys
from sklearn.model_selection import train_test_split

# data storage
data_train = "C:/Users/Manish/projects/tiya/scienceFair-2019/data/rawdata_exp_1024_train.dat"
data_test = "C:/Users/Manish/projects/tiya/scienceFair-2019/data/rawdata_exp_1024_test.dat"

# forcefully run the following
forceRun = True

if not os.path.isfile(data_train) or forceRun:
    # reading information on csv file
    tr = pd.read_csv('C:/Users/Manish/projects/tiya/scienceFair-2019/data/stage_2_train_labels.csv')
    # calculating more information for location of pneumonia
    tr['aspect_ratio'] = (tr['width'] / tr['height'])
    tr['area'] = tr['width'] * tr['height']

    # getting info from the images
    def get_info(patientId, tr, root_dir='C:/Users/Manish/projects/tiya/scienceFair-2019/data/stage_2_train_images/'):

        # reading all files (images)
        file_name = os.path.join(root_dir, f'{patientId}.dcm')
        dcm_data = pydicom.read_file(file_name)

        # getting locations of pneumonia on images (boxes)
        a = tr.loc[lambda tr: tr.patientId==patientId, :]
        boxes = []
        for i in range(len(a)):
            boxes.append((a.iloc[i]['x'],a.iloc[i]['y'],a.iloc[i]['width'],a.iloc[i]['height']))

        # resizing all images into 1024, 1024
        dcm_pixels = dcm_data.pixel_array

        # returning all information
        return {'age': int(dcm_data.PatientAge),
                'gender': dcm_data.PatientSex,
                'id': os.path.basename(file_name).split('.')[0],
                'pixel_spacing': float(dcm_data.PixelSpacing[0]),
                'boxes':boxes,
                'Modality':dcm_data.Modality,
                'pixel_data': dcm_pixels}

    # store all info into result list
    patient_ids = list(tr.patientId.unique())
    result = []
    for i in patient_ids:
        result.append(get_info(i, tr))

    demo = pd.DataFrame(result)
    demo['gender'] = demo['gender'].astype('category')
    demo['age'] = demo['age'].astype(int)

    # combining tr and demo
    tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left').drop(columns=['id','x','y','width','height']))
    tf = tr.drop_duplicates(subset=['patientId'])
    print (sys.getsizeof(tf))

    # splitting train and test
    train, test = train_test_split(tf, test_size=0.2)

    # saving train into data file
    with open(data_train, "wb") as fout:
        print ("Saving data file into %s" %(data_train))
        pickle.dump((train),fout)

    # saving test into data file
    with open(data_test, "wb") as fout:
        print ("Saving data file into %s" %(data_test))
        pickle.dump((test),fout)
else:
    # getting train set from data file
    print("Reading data from file %s" % (data_train))
    with open(data_train, "rb") as fin:
        train = pickle.load(fin)

    # getting test set from data file
    print("Reading data from file %s" % (data_test))
    with open(data_test, "rb") as fin:
        test = pickle.load(fin)