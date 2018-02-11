"""
Using HRV analysis summary to predict OSA Severity Level (based on AHI value)
Mild    : AHI < 5
Normal  : 5 <= AHI < 30
Severe  : 30 <= AHI

implemented by Nannapas Banluesombatkul
"""

import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import math


# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 512 # set to 2389 to not using minibatch
display_step = 100
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412 # column index of AHI value in file
train_percent = 0.8 # percent of data using for train

# Network Parameters
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 512 # 2nd layer number of neurons
num_input = 19 # HRV summary column (features)
num_classes = 3 # AHI total classes

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# get train dataset
def get_data(trainX_fname, trainY_fname, labeled_column):

    dataX = genfromtxt(trainX_fname, delimiter=',', skip_header=1, dtype='string')
    patient_id = dataX[:,0]
    dataX = dataX[:,1:]

    # select only AHI value which contained in hrv summary data
    dataY = genfromtxt(trainY_fname, delimiter=',', skip_header=1, dtype='string')
    dataY = dataY[np.isin(dataY[:,0], patient_id), labeled_column].astype('int')
    tmp = np.zeros(shape=(len(dataY), num_classes), dtype='int')
    tmp[dataY < 5] = [1, 0, 0]
    tmp[dataY >= 30] = [0, 0, 1]
    tmp[dataY >= 5] = [0, 1, 0]

    return tf.convert_to_tensor(dataX, dtype=tf.float32),tf.convert_to_tensor(tmp, dtype=tf.int16)

# define get data ops
train_data_X, train_data_Y = get_data(trainX_fname=trainX, trainY_fname=trainY, 
                                        labeled_column=column)

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # devide data to train and test
    trX, trY = sess.run([train_data_X, train_data_Y])
    train_size = int(train_percent * len(trX))
    teX = trX[train_size:,]
    teY = trY[train_size:]
    trX = trX[:train_size]
    trY = trY[:train_size]
    batch_X, batch_Y = None, None

    
    for step in range(1, num_steps+1):

        if batch_size != len(trX):
            r = np.random.random_integers(len(trX)-1, size=batch_size)
            batch_X = trX[r,:]
            batch_Y = trY[r]
        else:
            batch_X = trX
            batch_Y = trY
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_X, Y: batch_Y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_X,
                                                                 Y: batch_Y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        
    print("Optimization Finished!")
    
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={  X: teX,
                                        Y: teY}))
    