from __future__ import print_function

import tensorflow as tf
import numpy
from numpy import genfromtxt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412
num_input = 19

# # Training Data
# train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# n_samples = train_X.shape[0]

# get train dataset
def get_data(trainX_fname, trainY_fname, labeled_column):

    dataX = genfromtxt(trainX_fname, delimiter=',', skip_header=1, dtype='string')
    patient_id = dataX[:,0]
    dataX = dataX[:,1:]

    # select only AHI value which contained in hrv summary data
    dataY = genfromtxt(trainY_fname, delimiter=',', skip_header=1, dtype='string')
    dataY = dataY[numpy.isin(dataY[:,0], patient_id), labeled_column].astype('int')

    return tf.convert_to_tensor(dataX, dtype=tf.float32),tf.convert_to_tensor(dataY, dtype=tf.int16)

# define get data ops
train_X, train_Y = get_data(trainX_fname=trainX, trainY_fname=trainY, 
                                        labeled_column=column)
n_samples = train_X.get_shape().as_list()[0]

# tf Graph Input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None])

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
W1 = tf.Variable(rng.randn(), name="weight")
b1 = tf.Variable(rng.randn(), name="bias")

""" train """
# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

""" check accuracy """
# Construct a linear model
pred1 = tf.add(tf.multiply(X, W1), b1)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred1-Y, 2))/(2*n_samples)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    train_X, train_Y = sess.run([train_X, train_Y])

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("shape:", len(train_X), len(train_Y))
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # # Testing example, as requested (Issue #2)
    # test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    # test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    # print("Testing... (Mean square loss Comparison)")
    # testing_cost = sess.run(
    #     tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
    #     feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    # print("Testing cost=", testing_cost)
    # print("Absolute mean square loss difference:", abs(
    #     training_cost - testing_cost))