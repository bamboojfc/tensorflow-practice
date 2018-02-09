"""
Using HRV analysis summary to predict OSA Severity Level (based on AHI value)
Mild    : AHI < 5
Normal  : 5 <= AHI < 15
Severe  : 15 <= AHI

implemented by Nannapas Banluesombatkul
"""

import tensorflow as tf
from numpy import genfromtxt
import numpy as np

## file lists ##
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412 #column index of AHI value in file

## initialize global variables ##
trPid = []

def get_data(trainX_fname, trainY_fname, labeled_column):
    
    dataX = genfromtxt(trainX_fname, delimiter=',', skip_header=1, dtype='string')
    patient_id = dataX[:,0]
    dataX = dataX[:,1:]

    # select only AHI value which contained in hrv summary data
    dataY = genfromtxt(trainY_fname, delimiter=',', skip_header=1, dtype='string')
    dataY = dataY[np.isin(dataY[:,0], patient_id), labeled_column]

    return tf.convert_to_tensor(dataX, dtype=tf.float32),tf.convert_to_tensor(dataY, dtype=tf.int16)

train_data_X, train_data_Y = get_data(trainX_fname=trainX, trainY_fname=trainY, labeled_column=column)


with tf.Session() as sess:
    trX, trY = sess.run([train_data_X, train_data_Y])
    print 'trX', len(trX), trX
    print 'trY', len(trY), trY
    