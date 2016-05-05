# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data

print("finished import")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("finished assignment")
