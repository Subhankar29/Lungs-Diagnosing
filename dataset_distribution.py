import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_Entry_2017.csv')
X = dataset.iloc[24999:34999, 0:1].values
y = dataset.iloc[24999:34999, 1:2].values
X = pd.DataFrame(X)
y = pd.DataFrame(y)

import re
list1 = []
for i in range(0, 10000):
    review = re.sub('[^a-zA-Z_]', ' ', dataset['Finding Labels'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    review=review.split(' ', 1)[0]
    list1.append(review)
y = pd.DataFrame(list1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

X_train= X_train[0].tolist()
X_test= X_test[0].tolist()
y_train= y_train[0].tolist()
y_test= y_test[0].tolist()

path1 = 'images'

for i in range(0,1500) :
    im = Image.open(path1 + '\\' + X_test[i])   
    if y_test[i] == 'pneumothorax':
        im.save('Dataset\Test set\pneumothorax' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'cardiomegaly':
        im.save('Dataset\Test set\cardiomegaly' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'edema':
        im.save('Dataset\Test set\edema' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'effusion':
        im.save('Dataset\Test set\effusion' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'fibrosis':
        im.save('Dataset\Test set\fibrosis' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'atelectasis':
        im.save('Dataset\Test set\atelectasis' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'hernia':
        im.save('Dataset\Test set\hernia' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'infiltration':
        im.save('Dataset\Test set\infiltration' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'emphysema':
        im.save('Dataset\Test set\emphysema' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'consolidation':
        im.save('Dataset\Test set\consolidation' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'mass':
        im.save('Dataset\Test set\mass' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'pneumonia':
        im.save('Dataset\Test set\pneumonia' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'pleural_thickening':
        im.save('Dataset\Test set\pleural_thickening' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'no_finding':
        im.save('Dataset\Test set\no_finding' +'\\' +  X_test[i] , "JPEG")
    elif y_test[i] == 'nodule':
        im.save('Dataset\Test set\nodule' +'\\' +  X_test[i] , "JPEG")


for i in range(0,8500) :
    im = Image.open(path1 + '\\' + X_train[i])   
    if y_train[i] == 'pneumothorax':
        im.save('Dataset\Training set\pneumothorax' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'cardiomegaly':
        im.save('Dataset\Training set\cardiomegaly' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'edema':
        im.save('Dataset\Training set\edema' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'effusion':
        im.save('Dataset\Training set\effusion' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'fibrosis':
        im.save('Dataset\Training set\fibrosis' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'atelectasis':
        im.save('Dataset\Training set\atelectasis' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'hernia':
        im.save('Dataset\Training set\hernia' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'infiltration':
        im.save('Dataset\Training set\infiltration' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'emphysema':
        im.save('Dataset\Training set\emphysema' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'consolidation':
        im.save('Dataset\Training set\consolidation' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'mass':
        im.save('Dataset\Training set\mass' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'pneumonia':
        im.save('Dataset\Training set\pneumonia' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'pleural_thickening':
        im.save('Dataset\Training set\pleural_thickening' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'no_finding':
        im.save('Dataset\Training set\no_finding' +'\\' +  X_train[i] , "JPEG")
    elif y_train[i] == 'nodule':
        im.save('Dataset\Training set\nodule' +'\\' +  X_train[i] , "JPEG")
