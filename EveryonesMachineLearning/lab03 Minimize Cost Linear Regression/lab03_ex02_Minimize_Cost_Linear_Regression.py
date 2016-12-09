# https://www.youtube.com/watch?v=pHPmzTQ_e2o&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=6#t=59.448354
import tensorflow as tf

# training data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Simplified H = WX
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost(W, b)

# Minimize alpha = 0.1
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(1000):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    if 0 == (step % 200):
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
# end of training

# running
print sess.run(hypothesis, feed_dict={X: 5})
print sess.run(hypothesis, feed_dict={X: 2.5})
