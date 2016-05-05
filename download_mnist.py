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

print('dir(mnist) = ' + str(dir(mnist)))
print('dir(mnist.train) = ' + str(dir(mnist.train)))
print('mnist.train = ' + str(mnist.train))
print('mnist.train.images = ' + str(mnist.train.images))
print('mnist.train.images.shape = ' + str(mnist.train.images.shape))
print('mnist.train.labels = ' + str(mnist.train.labels))
print('mnist.train.labels.shape = ' + str(mnist.train.labels.shape))
print('mnist.test = ' + str(mnist.test))
print('mnist.validation = ' + str(mnist.validation))
