# https://www.youtube.com/watch?v=4HrSxpi3IAM&index=3&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
import tensorflow as tf

# training data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Variables : so that W, b can be updated
# initialize with a random number
# actual initialization happens @ tf.initialize_all_variables()
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b  # H(x) = Wx + b

print("hypothesis = %s" % hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost(W, b)

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
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    # intermediate report
    if 0 == step % 200:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)
# end of training

# running
print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.55})
