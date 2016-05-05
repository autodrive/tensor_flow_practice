# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    global mnist
    success = False
    while not success:

        try:
            print("trying download")
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            print("finished assignment")
        except IOError:
            success = False
        else:
            # if there was no error
            success = True
    return mnist


mnist = load_data()

# present information
print('dir(mnist) = ' + str(dir(mnist)))
print('dir(mnist.train) = ' + str(dir(mnist.train)))
print('mnist.train = ' + str(mnist.train))
print('mnist.train.images = ' + str(mnist.train.images))
print('mnist.train.images.shape = ' + str(mnist.train.images.shape))
print('mnist.train.labels = ' + str(mnist.train.labels))
print('mnist.train.labels.shape = ' + str(mnist.train.labels.shape))
print('mnist.test = ' + str(mnist.test))
print('mnist.validation = ' + str(mnist.validation))

# trying softmax regression
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

# Weight
W = tf.Variable(tf.zeros([mnist.train.images.shape[1], mnist.train.labels.shape[1]]))

# bias
b = tf.Variable(tf.zeros([mnist.train.labels.shape[1]]))

# model
# y = W x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

print('y = ' + str(y))
