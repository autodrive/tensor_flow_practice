import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)
c = a + b
print c
print sess.run(c)
