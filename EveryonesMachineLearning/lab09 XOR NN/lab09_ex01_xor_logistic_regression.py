# https://www.youtube.com/watch?v=9i7FBbcZPMA&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=24#t=34s
import numpy as np
import tensorflow as tf

# training data
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# [1, len(x_data)] because we may not know the number of matrices
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))  # H(x) = Wx + b
# cost function
# because Y value will be either zero or one
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.01)  # learning rate alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

# Launch the graph.
# https://www.youtube.com/watch?v=9i7FBbcZPMA&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=24#t=1m42s
with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in range(1001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        # intermediate report
        if 0 == step % 200:
            print("%s %s %s" % (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        sess.run(
            [hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
            feed_dict={X: x_data, Y: y_data}
        )
    )
    print("%s %s" % ("Accuracy:", accuracy.eval({X: x_data, Y: y_data})))
    # accuracy would not be high
