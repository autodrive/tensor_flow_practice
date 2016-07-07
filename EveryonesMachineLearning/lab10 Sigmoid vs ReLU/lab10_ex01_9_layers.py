# https://www.youtube.com/watch?v=cKtg_fpw88c#t=3m50s
import os

import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

# input layer width = x_data.shape[1]
# hidden layer width = 5
# number of hidden layers = 9
# output layer width = 1

n_nodes_list = [x_data.shape[1]] + [5] * 9 + [1]
weights_list = []
weights_histograms_list = []
biases_list = []
biases_histograms_list = []
layers_list = []

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')
y_hist = tf.histogram_summary("y", Y)

# Input Layer
with tf.name_scope("layer1") as scope:
    L = X
layers_list.append(L)

# layers loop
for k, width in enumerate(n_nodes_list[:-1]):
    # Deep network configuration.: Use more layers.
    W = tf.Variable(tf.random_uniform([n_nodes_list[k], n_nodes_list[k + 1]],
                                      -1.0, 1.0),
                    name='weight%d' % (k + 1))
    # Add histogram
    w_hist = tf.histogram_summary("weights%d" % (k + 1), W)
    b = tf.Variable(tf.zeros([n_nodes_list[k + 1]]), name="bias%d" % (k + 1))
    b_hist = tf.histogram_summary("biases%d" % (k + 1), b)

    # Hypotheses
    with tf.name_scope("layer%d" % (k + 2)) as scope:
        L = tf.sigmoid(tf.matmul(layers_list[-1], W) + b)

    weights_list.append(W)
    weights_histograms_list.append(w_hist)
    biases_list.append(b)
    biases_histograms_list.append(b_hist)
    layers_list.append(L)

# output layer
with tf.name_scope("output") as scope:
    hypothesis = layers_list[-1]

# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)

# Minimize cost.
a = tf.Variable(0.1)
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.initialize_all_variables()

log_dir = os.path.join(os.curdir, 'logs', 'xor_logs')

# Launch the graph
with tf.Session() as sess:
    # tensorboard merge
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph)

    sess.run(init)

    # Run graph.
    for step in range(8750 + 1):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))

import sys

if 2 <= len(sys.argv):
    tensorboard = sys.argv[1]
    os.system("%s --logdir=%s" % (tensorboard, log_dir))
# http://0.0.0.0:6006
