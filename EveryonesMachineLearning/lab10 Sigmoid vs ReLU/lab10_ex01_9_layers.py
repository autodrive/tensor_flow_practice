# https://www.youtube.com/watch?v=cKtg_fpw88c#t=3m50s

import os

import numpy as np
import tensorflow as tf

xy = np.loadtxt('xor_dataset.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

n_nodes_list = [x_data.shape]

# Deep network configuration.: Use more layers.
W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight3')
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight4')
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight5')
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight6')
W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight7')
W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight8')
W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight9')
W10 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight10')
W11 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight11')

b1 = tf.Variable(tf.zeros([5]), name="bias1")
b2 = tf.Variable(tf.zeros([5]), name="bias2")
b3 = tf.Variable(tf.zeros([5]), name="bias3")
b4 = tf.Variable(tf.zeros([5]), name="bias4")
b5 = tf.Variable(tf.zeros([5]), name="bias5")
b6 = tf.Variable(tf.zeros([5]), name="bias6")
b7 = tf.Variable(tf.zeros([5]), name="bias7")
b8 = tf.Variable(tf.zeros([5]), name="bias8")
b9 = tf.Variable(tf.zeros([5]), name="bias9")
b10 = tf.Variable(tf.zeros([5]), name="bias10")
b11 = tf.Variable(tf.zeros([1]), name="bias11")

# Add histogram
w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W2)
w3_hist = tf.histogram_summary("weights3", W3)
w4_hist = tf.histogram_summary("weights4", W4)
w5_hist = tf.histogram_summary("weights5", W5)
w6_hist = tf.histogram_summary("weights6", W6)
w7_hist = tf.histogram_summary("weights7", W7)
w8_hist = tf.histogram_summary("weights8", W8)
w9_hist = tf.histogram_summary("weights9", W9)
w10_hist = tf.histogram_summary("weights10", W10)
w11_hist = tf.histogram_summary("weights11", W11)

b1_hist = tf.histogram_summary("biases1", b1)
b2_hist = tf.histogram_summary("biases2", b2)
b3_hist = tf.histogram_summary("biases3", b3)
b4_hist = tf.histogram_summary("biases4", b4)
b5_hist = tf.histogram_summary("biases5", b5)
b6_hist = tf.histogram_summary("biases6", b6)
b7_hist = tf.histogram_summary("biases7", b7)
b8_hist = tf.histogram_summary("biases8", b8)
b9_hist = tf.histogram_summary("biases9", b9)
b10_hist = tf.histogram_summary("biases10", b10)
b11_hist = tf.histogram_summary("biases11", b11)

y_hist = tf.histogram_summary("y", Y)

# Hypotheses
# input layer
with tf.name_scope("layer1") as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# hidden layers
with tf.name_scope("layer2") as scope:
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
with tf.name_scope("layer3") as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
with tf.name_scope("layer4") as scope:
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
with tf.name_scope("layer5") as scope:
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
with tf.name_scope("layer6") as scope:
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
with tf.name_scope("layer7") as scope:
    L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
with tf.name_scope("layer8") as scope:
    L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
with tf.name_scope("layer9") as scope:
    L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)
with tf.name_scope("layer10") as scope:
    L10 = tf.nn.relu(tf.matmul(L9, W10) + b10)

# output layer
with tf.name_scope("last") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10, W11) + b11)

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
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
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
