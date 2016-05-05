# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data

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
