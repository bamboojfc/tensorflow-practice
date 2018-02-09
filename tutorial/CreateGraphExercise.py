import tensorflow as tf

b = tf.constant(5)
a = tf.constant(3)

d = a + b
c = a * b
f = d + c
e = d - c
g = f / e

sess = tf.Session()
print b                     # b is a Tensor object
print sess.run(b)           # this returns an actual value of b
print 'g =', sess.run(g)    # this returns an actual value of g by walk through all nodes which g depends on (all)