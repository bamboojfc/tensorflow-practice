import tensorflow as tf
import numpy as np


# parameters
num_features = 19 # HRV summary data
num_classes = 3
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412 # column index of AHI value in file
train_percent = 0.8 # percent of data using for train

# import data
def get_data(trainX_fname, trainY_fname, labeled_column):

    dataX = np.genfromtxt(trainX_fname, delimiter=',', skip_header=1, dtype='string')
    patient_id = dataX[:,0]
    dataX = dataX[:,1:]

    # select only AHI value which contained in hrv summary data
    dataY = np.genfromtxt(trainY_fname, delimiter=',', skip_header=1, dtype='string')
    dataY = dataY[np.isin(dataY[:,0], patient_id), labeled_column].astype('int')
    tmp = np.zeros(shape=(len(dataY)), dtype='int')
    tmp[dataY < 5] = 1
    tmp[dataY >= 30] = 3
    tmp[dataY >= 5] = 2

    return tf.convert_to_tensor(dataX, dtype=tf.float32),tf.convert_to_tensor(tmp, dtype=tf.int16)

# define get data ops
train_data_X, train_data_Y = get_data(trainX_fname=trainX, trainY_fname=trainY, 
                                        labeled_column=column)

# tf Graph Input
xtr = tf.placeholder("float", [None, num_features])
xte = tf.placeholder("float", [num_features])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.argmin(distance)

accuracy = 0.

# Start training
with tf.Session() as sess:
    
    Xtr, Ytr = sess.run([train_data_X, train_data_Y])
    train_size = int(train_percent * len(Xtr))
    Xte = Xtr[train_size:,]
    Yte = Ytr[train_size:]
    Xtr = Xtr[:train_size]
    Ytr = Ytr[:train_size]

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print "Test", i, "Mean distance:", sess.run(distance, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print "Prediction:", Ytr[nn_index], \
            "True Class:", Yte[i]
        # Calculate accuracy
        if Ytr[nn_index] == Yte[i]:
            accuracy += 1./len(Xte)
    print "Done!"
    print "Accuracy:", accuracy