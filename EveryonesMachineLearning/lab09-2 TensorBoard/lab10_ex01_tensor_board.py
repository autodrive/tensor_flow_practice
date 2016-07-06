# Name varialbes    https://www.youtube.com/watch?v=eDKxY5Z5dVQ#t=4m:47s
# Grouping          https://www.youtube.com/watch?v=eDKxY5Z5dVQ#t=5m:06s
# Histogram Summary https://www.youtube.com/watch?v=eDKxY5Z5dVQ#t=5m:48s
# Merge             https://www.youtube.com/watch?v=eDKxY5Z5dVQ#t=6m:54s

import os

import numpy as np
import tensorflow as tf

# training data
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

# variables
X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

# number of inputs and outputs of each layer
n_layer001_input = 2
n_layer001_output = 5
n_layer002_input = n_layer001_output
n_layer002_output = 4
n_layer003_input = n_layer002_output
n_layer003_output = 1

W1 = tf.Variable(tf.random_uniform([n_layer001_input, n_layer001_output], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([n_layer002_input, n_layer002_output], -1.0, 1.0), name='Weight2')
W3 = tf.Variable(tf.random_uniform([n_layer003_input, n_layer003_output], -1.0, 1.0), name='Weight3')

# TODO : try name_scope("weights") or "biases"
w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W2)
w3_hist = tf.histogram_summary("weights3", W3)

b1 = tf.Variable(tf.zeros([n_layer001_output]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_layer002_output]), name="Bias2")
b3 = tf.Variable(tf.zeros([n_layer003_output]), name="Bias3")

# TODO : try name_scope("weights") or "biases"
b1_hist = tf.histogram_summary("biases1", W1)
b2_hist = tf.histogram_summary("biases2", W2)
b3_hist = tf.histogram_summary("biases3", W3)

y_hist = tf.histogram_summary("y", Y)

# hypothesis
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
with tf.name_scope("layer4") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# cost function
# because Y value will be either zero or one
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)

# Minimize
a = tf.Variable(0.1)  # learning rate alpha
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

# Launch the graph.
# https://www.youtube.com/watch?v=9i7FBbcZPMA&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=24#t=1m42s
with tf.Session() as sess:
    sess.run(init)

    # tensorboard --logdir=./logs/xor_logs

    # merge summaries
    # all summaries go here
    merged = tf.merge_all_summaries()

    # create writer as if opening a file
    # sess.graph_def -> sess.graph because .graph_def depricated
    writer = tf.train.SummaryWriter(os.path.join(os.curdir, 'logs', 'xor_logs'),
                                    sess.graph)
    # "writer" is similar to a file?

    # Fit the line.
    for step in range(8001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        # intermediate report
        if 0 == step % 1000:
            print("%s %s %s %s" % (
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                sess.run(W1),
                sess.run(W2)))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        sess.run(
            [hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
            feed_dict={X: x_data, Y: y_data}
        )
    )
    print("%s %s" % ("Accuracy:", accuracy.eval({X: x_data, Y: y_data})))
    # accuracy would be higher
