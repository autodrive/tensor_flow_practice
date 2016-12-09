# https://www.youtube.com/watch?v=4HrSxpi3IAM&index=3&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
import tensorflow as tf

# training data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Variables : so that W, b can be updated
# initialize with a random number
# actual initialization happens @ tf.initialize_all_variables()
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# expected result: W -> 1, b -> 0
hypothesis = W * x_data + b  # H(x) = Wx + b

print("hypothesis = %s" % hypothesis)

# cost = sum( {(W * xi + b) - yi} ** 2 ) / n
cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost(W, b)

print("cost = %s" % cost)

# a: learning rate
# magic? black box?
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

print("begin train ".ljust(40, '='))
print(train)
print("end train ".ljust(40, '='))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    # intermediate report
    if 0 == step % 200:
        print step, sess.run(cost), sess.run(W), sess.run(b)
