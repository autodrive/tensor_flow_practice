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

hypothesis = W * x_data + b  # H(x) = Wx + b

print("hypothesis = %s" % hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost(W, b)

# magic? black box?
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    # intermediate report
    if 0 == step % 20:
        print step, sess.run(cost), sess.run(W), sess.run(b)
