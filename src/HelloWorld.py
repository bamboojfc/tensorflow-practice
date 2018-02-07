import tensorflow as tf

hello = tf.constant('Hello, TensowFlow!')

sess = tf.Session()

print sess.run(hello)