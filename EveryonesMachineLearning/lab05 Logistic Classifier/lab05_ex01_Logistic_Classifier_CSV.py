# https://www.youtube.com/watch?v=t7Y9luCNzzE&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=12#t=138s
import tensorflow as tf
import numpy as np

# training data
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# [1, len(x_data)] because we may not know the number of matrices
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))  # H(x) = Wx + b
# cost function
# because Y value will be either zero or one
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1)  # learning rate alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    # intermediate report
    if 0 == step % 20:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
