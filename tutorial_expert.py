# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
