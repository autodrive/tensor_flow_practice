# https://www.youtube.com/watch?v=cKtg_fpw88c#t=3m50s
import os

import numpy as np
import tensorflow as tf


def main():
    xy = np.loadtxt('xor_dataset.txt', unpack=True)

    x_data = np.transpose(xy[0:-1])
    y_data = np.reshape(xy[-1], (4, 1))

    # input layer width = x_data.shape[1]
    # hidden layer width = 5
    # number of hidden layers = 9
    # output layer width = 1

    b_weights_histogram = False
    b_biases_histogram = False

    n_nodes_list = [x_data.shape[1]] + [5] * 9 + [1]
    weights_list = []
    biases_list = []
    layers_list = []

    weights_histograms_list = []
    biases_histograms_list = []

    X = tf.placeholder(tf.float32, name='X-input')
    Y = tf.placeholder(tf.float32, name='Y-input')
    y_hist = tf.histogram_summary("y", Y)

    # Input Layer
    with tf.name_scope("input_layer") as scope:
        layer = X
        layers_list.append(layer)

    last_layer = layer
    last_width = x_data.shape[1]

    # layers loop
    for k, width in enumerate(n_nodes_list[1:]):
        # Deep network configuration.: Use more layers.
        weight = tf.Variable(tf.random_uniform([last_width, width],
                                               -1.0, 1.0),
                             name='weight%d' % (k + 1))
        bias = tf.Variable(tf.zeros([n_nodes_list[k + 1]]), name="bias%d" % (k + 1))

        with tf.name_scope("layer%d" % (k + 1)) as scope:
            layer = tf.sigmoid(tf.matmul(last_layer, weight) + bias)

        weights_list.append(weight)
        biases_list.append(bias)
        layers_list.append(layer)

        last_layer = layer
        last_width = width

        # Add histogram

        if b_weights_histogram:
            w_hist = tf.histogram_summary("weights%d" % (k + 1), weight)
            weights_histograms_list.append(w_hist)
        if b_biases_histogram:
            b_hist = tf.histogram_summary("biases%d" % (k + 1), bias)
            biases_histograms_list.append(b_hist)

    # output layer
    with tf.name_scope("output_layer") as scope:
        hypothesis = layers_list[-1]

    cost = design_cost_function(Y, hypothesis)

    train = design_optimizer(cost)

    log_dir = os.path.join(os.curdir, 'logs', 'xor_logs')

    run_graph(X, Y, hypothesis, train, x_data, y_data)

    run_tensorboard(log_dir)


def design_cost_function(Y, hypothesis):
    # Cost function
    with tf.name_scope("cost") as scope:
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
        cost_summ = tf.scalar_summary("cost", cost)
    return cost


def design_optimizer(cost):
    # Minimize cost.
    a = tf.Variable(0.1)
    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(cost)
    return train


def run_graph(X, Y, hypothesis, train, x_data, y_data):
    # Initializa all variables.
    init = tf.initialize_all_variables()
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


def run_tensorboard(log_dir):
    import sys
    if 2 <= len(sys.argv):
        tensorboard = sys.argv[1]
        os.system("%s --logdir=%s" % (tensorboard, log_dir))
        # http://0.0.0.0:6006


if __name__ == '__main__':
    main()
