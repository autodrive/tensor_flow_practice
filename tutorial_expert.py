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

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# training
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
print("correct_prediction =" + str(correct_prediction))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
