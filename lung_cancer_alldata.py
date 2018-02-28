

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import pandas as pd

TRAIN_DIR = './lungs_images'
TEST_DIR = './test1'
IMG_SIZE = 50
LR = 1e-2

MODEL_NAME = 'lung_cancer-{}-{}.model'.format(LR, 'lungs-basic-test')
labels = {1:'Atelectasis', 2: 'Cardiomegaly', 3: 'Effusion', 4: 'Infiltration', 5: 'Mass', 6: 'Nodule', 7: 'Pneumonia', 8:
'Pneumothorax', 9: 'Consolidation', 10: 'Edema', 11: 'Emphysema', 12: 'Fibrosis', 13:
'Pleural_Thickening', 14: 'Hernia',15:'No Finding'}
labels = {v: k for k, v in labels.iteritems()}






import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 15, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')






model.load(MODEL_NAME)
print('model loaded!')







def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data1.npy', testing_data)
    return testing_data


# In[18]:




# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:

# test_data = np.load('test_data.npy')



for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]





# model_out is the array of prob 
