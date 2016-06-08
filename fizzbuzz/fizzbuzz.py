# coding: utf8
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
# https://tensorflowkorea.wordpress.com/2016/05/24/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf


def binary_encode(i, num_digits):
    """

    Args:
        i:
        num_digits:

    Returns:

    >>> binary_encode(0, 0)
    []
    >>> binary_encode(0, 1)
    [0]
    >>> binary_encode(0, 2)
    [0 0]
    >>> binary_encode(1, 1)
    [1]
    >>> binary_encode(1, 2)
    [1 0]
    >>> binary_encode(1, 3)
    [1 0 0]
    >>> binary_encode(2, 2)
    [0 1]
    >>> binary_encode(2, 3)
    [0 1 0]
    >>> binary_encode(2, 4)
    [0 1 0 0]
    >>> binary_encode(3, 3)
    [1 1 0]
    >>> binary_encode(3, 4)
    [1 1 0 0]
    >>> binary_encode(3, 5)
    [1 1 0 0 0]
    >>> binary_encode(4, 4)
    [0 0 1 0]
    >>> binary_encode(4, 5)
    [0 0 1 0 0]
    >>> binary_encode(4, 6)
    [0 0 1 0 0 0]

    """
    return np.array([i >> d & 1 for d in range(num_digits)])


def fizz_buzz_encode(i):
    """

    Args:
        i:

    Returns:
    >>> fizz_buzz_encode(0)
    array([0, 0, 0, 1])
    .>>> fizz_buzz_encode(3)
    array([0, 1, 0, 0])
    >>> fizz_buzz_encode(5)
    array([0, 0, 1, 0])
    >>> fizz_buzz_encode(15)
    array([0, 0, 0, 1])
    """
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    # one hidden layer
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def main():
    # generate some training data
    NUM_DIGITS = 10
    trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    # number of hidden layers. may change later
    NUM_HIDDEN = 100

    # input variables
    X = tf.placeholder("float", [None, NUM_DIGITS])
    Y = tf.placeholder("float", [None, 4])

    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 4])

    py_x = model(X, w_h, w_o)

    # cross entropy as cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

    # training objective is to minimize it
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    # the prediction will just be the largest output
    predict_op = tf.argmax(py_x, 1)


if __name__ == '__main__':
    main()
