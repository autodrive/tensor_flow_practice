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
