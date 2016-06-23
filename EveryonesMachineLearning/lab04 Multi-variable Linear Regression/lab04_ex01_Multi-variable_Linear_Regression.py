# https://www.youtube.com/watch?v=iEaVR1N8EEk&index=9&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm#t=70s
# https://www.youtube.com/watch?v=iEaVR1N8EEk&index=9&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm#t=180s
import tensorflow as tf

# training data
x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data = [1, 2, 3, 4, 5]

# Variables : so that W, b can be updated
# initialize with a random number
# actual initialization happens @ tf.initialize_all_variables()
W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1 * x1_data + W2 * x2_data + b  # H(x) = Wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost(W, b)

# magic? black box?
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    # intermediate report
    if 0 == step % 20:
        print step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)
