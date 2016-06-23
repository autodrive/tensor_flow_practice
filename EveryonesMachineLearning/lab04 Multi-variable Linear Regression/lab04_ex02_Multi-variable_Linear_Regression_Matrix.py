# https://www.youtube.com/watch?v=iEaVR1N8EEk&index=9&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm#t=386s
import tensorflow as tf

# training data
x_data = [[0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

# Variables : so that W, b can be updated
# initialize with a random number
# actual initialization happens @ tf.initialize_all_variables()
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data) + b  # H(x) = Wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost(W, b)

# Minimize
a = tf.Variable(0.1)  # learning rate alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    # intermediate report
    if 0 == step % 20:
        print step, sess.run(cost), sess.run(W), sess.run(b)
