# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# interactive session
import tensorflow as tf

sess = tf.InteractiveSession()

# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# run session
sess.run(tf.initialize_all_variables())

# model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
