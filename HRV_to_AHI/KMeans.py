import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Parameters
num_steps = 10 # Total steps to train
k = 6 # The number of clusters algm
num_classes = 3 # Actual class numbers
num_features = 19 # HRV summary data
trainX = '../data/mros-visit1-hrv-summary-0.3.0.csv'
trainY = '../data/mros-visit1-dataset-0.3.0.csv'
column = 412 # column index of AHI value in file
train_percent = 0.8 # percent of data using for train

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,
train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# get train dataset
def get_data(trainX_fname, trainY_fname, labeled_column):

    dataX = np.genfromtxt(trainX_fname, delimiter=',', skip_header=1, dtype='string')
    patient_id = dataX[:,0]
    dataX = dataX[:,1:]

    # select only AHI value which contained in hrv summary data
    dataY = np.genfromtxt(trainY_fname, delimiter=',', skip_header=1, dtype='string')
    dataY = dataY[np.isin(dataY[:,0], patient_id), labeled_column].astype('int')
    tmp = np.zeros(shape=(len(dataY), num_classes), dtype='int')
    tmp[dataY < 5] = [1, 0, 0]
    tmp[dataY >= 30] = [0, 0, 1]
    tmp[dataY >= 5] = [0, 1, 0]

    return tf.convert_to_tensor(dataX, dtype=tf.float32),tf.convert_to_tensor(tmp, dtype=tf.int16)


# define get data ops
train_data_X, train_data_Y = get_data(trainX_fname=trainX, trainY_fname=trainY, 
                                        labeled_column=column)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# devide train and test set
trX, trY = sess.run([train_data_X, train_data_Y])
train_size = int(train_percent * len(trX))
teX = trX[train_size:,]
teY = trY[train_size:]
trX = trX[:train_size]
trY = trY[:train_size]

# Run the initializer
sess.run(init_vars, feed_dict={X: trX, Y: trY})
sess.run(init_op, feed_dict={X: trX})

# Training
for i in range(1, num_steps + 1):
    sc, _, d, idx = sess.run([scores, train_op, avg_distance, cluster_idx],
                         feed_dict={X: trX})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))

for i in range(len(idx)):
    counts[idx[i]] += trY[i]

# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: teX, Y: teY}))