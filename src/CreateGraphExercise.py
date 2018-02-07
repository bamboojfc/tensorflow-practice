import tensorflow as tf

b = tf.constant(5)
a = tf.constant(3)

d = a + b
c = a * b
f = d + c
e = d - c
g = f / e

sess = tf.Session()
print 'g =', sess.run(g)