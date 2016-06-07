# coding: utf8
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
# https://tensorflowkorea.wordpress.com/2016/05/24/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf


def binary_encode(i, num_digits):
    return np.array([i >> d] & 1 for d in range(num_digits))
