import tensorflow as tf

# define a placeholder [a]
a = tf.placeholder(tf.int16)
# define another placeholder [b]
b = tf.placeholder(tf.int16)

# define a graph of [a] + [b]
add = tf.add(a, b)
# define a graph of [a] * [b]
mul = tf.mul(a, b)

# show graphs
print("add = %s" % add)
print("mul = %s" % mul)

# evaluate graphs
with tf.Session() as sess:
    input_to_graphs = {a: 2, b: 3}
    result_from_add_graph = sess.run(add, feed_dict=input_to_graphs)
    print ("Addition with variables: %i" % result_from_add_graph)

    result_from_mul_graph = sess.run(mul, feed_dict=input_to_graphs)
    print ("Multiplication with variables: %i" % result_from_mul_graph)
