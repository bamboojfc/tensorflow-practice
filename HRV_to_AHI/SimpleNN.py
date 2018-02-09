"""
Using HRV analysis summary to predict OSA Severity Level (based on AHI value)
Mild : AHI < 5
Normal : 5 <= AHI < 15
Severe : 15 <= AHI
"""

import tensorflow as tf
from numpy import genfromtxt

train = '../data/mros-visit1-hrv-summary-0.3.0.csv'

def get_data(filename):
    my_data = genfromtxt(filename, delimiter=',', skip_header=1, dtype='string')
    patient_id = my_data[:,0]
    my_data = my_data[:,1:]
    return tf.convert_to_tensor(my_data, dtype=tf.float32), tf.convert_to_tensor(patient_id)

train_data = get_data(train)

with tf.Session() as sess:
    train_d, train_pid = sess.run(train_data)
    print len(train_d), len(train_d[0])
    print train_d[0]
    print len(train_pid)
    print train_pid
    