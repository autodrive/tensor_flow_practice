import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

# form a node of constant [2]
a = tf.constant(2)
# form another node of constant [3]
b = tf.constant(3)
# define a relationship graph [c] = [a] + [b]
c = a + b
# show the graph
print (c)
# evaluate the graph
print (sess.run(c))
