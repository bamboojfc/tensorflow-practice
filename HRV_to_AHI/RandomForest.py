from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import numpy as np

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_features = 19
num_trees = 10
max_nodes = 2000
num_classes = 3
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412 # column index of AHI value in file
train_percent = 0.8 # percent of data using for train

# get train dataset
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

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.train.MonitoredSession()

# Run the initializer
sess.run(init_vars)

trX, trY = sess.run([train_data_X, train_data_Y])  
train_size = int(train_percent * len(trX))
teX = trX[train_size:,]
teY = trY[train_size:]
trX = trX[:train_size]
trY = trY[:train_size]  
batch_X, batch_Y = None, None

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    if batch_size != len(trX):
        r = np.random.random_integers(len(trX)-1, size=batch_size)
        batch_X = trX[r,:]
        batch_Y = trY[r]
    else:
        batch_X = trX
        batch_Y = trY

    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_X, Y: batch_Y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_X, Y: batch_Y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: teX, Y: teY}))