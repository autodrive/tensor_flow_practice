# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html
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

training_data_feature_size = mnist.train.images.shape[1]
number_of_answer_classes = mnist.train.labels.shape[1]

x = tf.placeholder(tf.float32, [None, training_data_feature_size])

# Weight
W = tf.Variable(tf.zeros([training_data_feature_size, number_of_answer_classes]))

# bias
b = tf.Variable(tf.zeros([number_of_answer_classes]))
# print('help(tf.Variable) =')
# help(tf.Variable)
'''
help(tf.Variable) =
Help on class Variable in module tensorflow.python.ops.variables:

class Variable(__builtin__.object)
 |  See the [Variables How To](../../how_tos/variables/index.md) for a high
 |  level overview.
 |
 |  A variable maintains state in the graph across calls to `run()`. You add a
 |  variable to the graph by constructing an instance of the class `Variable`.
 |
 |  The `Variable()` constructor requires an initial value for the variable,
 |  which can be a `Tensor` of any type and shape. The initial value defines the
 |  type and shape of the variable. After construction, the type and shape of
 |  the variable are fixed. The value can be changed using one of the assign
 |  methods.
 |
 |  If you want to change the shape of a variable later you have to use an
 |  `assign` Op with `validate_shape=False`.
 |
 |  Just like any `Tensor`, variables created with `Variable()` can be used as
 |  inputs for other Ops in the graph. Additionally, all the operators
 |  overloaded for the `Tensor` class are carried over to variables, so you can
 |  also add nodes to the graph by just doing arithmetic on variables.
 |
 |  ```python
 |  import tensorflow as tf
 |
 |  # Create a variable.
 |  w = tf.Variable(<initial-value>, name=<optional-name>)
 |
 |  # Use the variable in the graph like any Tensor.
 |  y = tf.matmul(w, ...another variable or tensor...)
 |
 |  # The overloaded operators are available too.
 |  z = tf.sigmoid(w + b)
 |
 |  # Assign a new value to the variable with `assign()` or a related method.
 |  w.assign(w + 1.0)
 |  w.assign_add(1.0)
 |  ```
 |
 |  When you launch the graph, variables have to be explicitly initialized before
 |  you can run Ops that use their value. You can initialize a variable by
 |  running its *initializer op*, restoring the variable from a save file, or
 |  simply running an `assign` Op that assigns a value to the variable. In fact,
 |  the variable *initializer op* is just an `assign` Op that assigns the
 |  variable's initial value to the variable itself.
 |
 |  ```python
 |  # Launch the graph in a session.
 |  with tf.Session() as sess:
 |      # Run the variable initializer.
 |      sess.run(w.initializer)
 |      # ...you now can run ops that use the value of 'w'...
 |  ```
 |
 |  The most common initialization pattern is to use the convenience function
 |  `initialize_all_variables()` to add an Op to the graph that initializes
 |  all the variables. You then run that Op after launching the graph.
 |
 |  ```python
 |  # Add an Op to initialize all variables.
 |  init_op = tf.initialize_all_variables()
 |
 |  # Launch the graph in a session.
 |  with tf.Session() as sess:
 |      # Run the Op that initializes all variables.
 |      sess.run(init_op)
 |      # ...you can now run any Op that uses variable values...
 |  ```
 |
 |  If you need to create a variable with an initial value dependent on another
 |  variable, use the other variable's `initialized_value()`. This ensures that
 |  variables are initialized in the right order.
 |
 |  All variables are automatically collected in the graph where they are
 |  created. By default, the constructor adds the new variable to the graph
 |  collection `GraphKeys.VARIABLES`. The convenience function
 |  `all_variables()` returns the contents of that collection.
 |
 |  When building a machine learning model it is often convenient to distinguish
 |  betwen variables holding the trainable model parameters and other variables
 |  such as a `global step` variable used to count training steps. To make this
 |  easier, the variable constructor supports a `trainable=<bool>` parameter. If
 |  `True`, the new variable is also added to the graph collection
 |  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
 |  `trainable_variables()` returns the contents of this collection. The
 |  various `Optimizer` classes use this collection as the default list of
 |  variables to optimize.
 |
 |
 |  Creating a variable.
 |
 |  @@__init__
 |  @@initialized_value
 |
 |  Changing a variable value.
 |
 |  @@assign
 |  @@assign_add
 |  @@assign_sub
 |  @@scatter_sub
 |  @@count_up_to
 |
 |  @@eval
 |
 |  Properties.
 |
 |  @@name
 |  @@dtype
 |  @@get_shape
 |  @@device
 |  @@initializer
 |  @@graph
 |  @@op
 |
 |  Methods defined here:
 |
 |  __abs__ lambda a
 |
 |  __add__ lambda a, b
 |
 |  __and__ lambda a, b
 |
 |  __div__ lambda a, b
 |
 |  __floordiv__ lambda a, b
 |
 |  __ge__ lambda a, b
 |
 |  __getitem__ lambda a, b
 |
 |  __gt__ lambda a, b
 |
 |  __init__(self, initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)
 |      Creates a new variable with value `initial_value`.
 |
 |      The new variable is added to the graph collections listed in `collections`,
 |      which defaults to `[GraphKeys.VARIABLES]`.
 |
 |      If `trainable` is `True` the variable is also added to the graph collection
 |      `GraphKeys.TRAINABLE_VARIABLES`.
 |
 |      This constructor creates both a `variable` Op and an `assign` Op to set the
 |      variable to its initial value.
 |
 |      Args:
 |        initial_value: A `Tensor`, or Python object convertible to a `Tensor`.
 |          The initial value for the Variable. Must have a shape specified unless
 |          `validate_shape` is set to False.
 |        trainable: If `True`, the default, also adds the variable to the graph
 |          collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
 |          the default list of variables to use by the `Optimizer` classes.
 |        collections: List of graph collections keys. The new variable is added to
 |          these collections. Defaults to `[GraphKeys.VARIABLES]`.
 |        validate_shape: If `False`, allows the variable to be initialized with a
 |          value of unknown shape. If `True`, the default, the shape of
 |          `initial_value` must be known.
 |        caching_device: Optional device string describing where the Variable
 |          should be cached for reading.  Defaults to the Variable's device.
 |          If not `None`, caches on another device.  Typical use is to cache
 |          on the device where the Ops using the Variable reside, to deduplicate
 |          copying through `Switch` and other conditional statements.
 |        name: Optional name for the variable. Defaults to `'Variable'` and gets
 |          uniquified automatically.
 |        variable_def: `VariableDef` protocol buffer. If not `None`, recreates
 |          the Variable object with its contents. `variable_def` and the other
 |          arguments are mutually exclusive.
 |        dtype: If set, initial_value will be converted to the given type.
 |          If `None`, either the datatype will be kept (if `initial_value` is
 |          a Tensor), or `convert_to_tensor` will decide.
 |
 |      Returns:
 |        A Variable.
 |
 |      Raises:
 |        ValueError: If both `variable_def` and initial_value are specified.
 |        ValueError: If the initial value is not specified, or does not have a
 |          shape and `validate_shape` is `True`.
 |
 |  __invert__ lambda a
 |
 |  __iter__(self)
 |      Dummy method to prevent iteration. Do not call.
 |
 |      NOTE(mrry): If we register __getitem__ as an overloaded operator,
 |      Python will valiantly attempt to iterate over the variable's Tensor from 0
 |      to infinity.  Declaring this method prevents this unintended behavior.
 |
 |      Raises:
 |        TypeError: when invoked.
 |
 |  __le__ lambda a, b
 |
 |  __lt__ lambda a, b
 |
 |  __mod__ lambda a, b
 |
 |  __mul__ lambda a, b
 |
 |  __neg__ lambda a
 |
 |  __or__ lambda a, b
 |
 |  __pow__ lambda a, b
 |
 |  __radd__ lambda a, b
 |
 |  __rand__ lambda a, b
 |
 |  __rdiv__ lambda a, b
 |
 |  __rfloordiv__ lambda a, b
 |
 |  __rmod__ lambda a, b
 |
 |  __rmul__ lambda a, b
 |
 |  __ror__ lambda a, b
 |
 |  __rpow__ lambda a, b
 |
 |  __rsub__ lambda a, b
 |
 |  __rtruediv__ lambda a, b
 |
 |  __rxor__ lambda a, b
 |
 |  __sub__ lambda a, b
 |
 |  __truediv__ lambda a, b
 |
 |  __xor__ lambda a, b
 |
 |  assign(self, value, use_locking=False)
 |      Assigns a new value to the variable.
 |
 |      This is essentially a shortcut for `assign(self, value)`.
 |
 |      Args:
 |        value: A `Tensor`. The new value for this variable.
 |        use_locking: If `True`, use locking during the assignment.
 |
 |      Returns:
 |        A `Tensor` that will hold the new value of this variable after
 |        the assignment has completed.
 |
 |  assign_add(self, delta, use_locking=False)
 |      Adds a value to this variable.
 |
 |       This is essentially a shortcut for `assign_add(self, delta)`.
 |
 |      Args:
 |        delta: A `Tensor`. The value to add to this variable.
 |        use_locking: If `True`, use locking during the operation.
 |
 |      Returns:
 |        A `Tensor` that will hold the new value of this variable after
 |        the addition has completed.
 |
 |  assign_sub(self, delta, use_locking=False)
 |      Subtracts a value from this variable.
 |
 |      This is essentially a shortcut for `assign_sub(self, delta)`.
 |
 |      Args:
 |        delta: A `Tensor`. The value to subtract from this variable.
 |        use_locking: If `True`, use locking during the operation.
 |
 |      Returns:
 |        A `Tensor` that will hold the new value of this variable after
 |        the subtraction has completed.
 |
 |  count_up_to(self, limit)
 |      Increments this variable until it reaches `limit`.
 |
 |      When that Op is run it tries to increment the variable by `1`. If
 |      incrementing the variable would bring it above `limit` then the Op raises
 |      the exception `OutOfRangeError`.
 |
 |      If no error is raised, the Op outputs the value of the variable before
 |      the increment.
 |
 |      This is essentially a shortcut for `count_up_to(self, limit)`.
 |
 |      Args:
 |        limit: value at which incrementing the variable raises an error.
 |
 |      Returns:
 |        A `Tensor` that will hold the variable value before the increment. If no
 |        other Op modifies this variable, the values produced will all be
 |        distinct.
 |
 |  eval(self, session=None)
 |      In a session, computes and returns the value of this variable.
 |
 |      This is not a graph construction method, it does not add ops to the graph.
 |
 |      This convenience method requires a session where the graph containing this
 |      variable has been launched. If no session is passed, the default session is
 |      used.  See the [Session class](../../api_docs/python/client.md#Session) for
 |      more information on launching a graph and on sessions.
 |
 |      ```python
 |      v = tf.Variable([1, 2])
 |      init = tf.initialize_all_variables()
 |
 |      with tf.Session() as sess:
 |          sess.run(init)
 |          # Usage passing the session explicitly.
 |          print(v.eval(sess))
 |          # Usage with the default session.  The 'with' block
 |          # above makes 'sess' the default session.
 |          print(v.eval())
 |      ```
 |
 |      Args:
 |        session: The session to use to evaluate this variable. If
 |          none, the default session is used.
 |
 |      Returns:
 |        A numpy `ndarray` with a copy of the value of this variable.
 |
 |  get_shape(self)
 |      The `TensorShape` of this variable.
 |
 |      Returns:
 |        A `TensorShape`.
 |
 |  initialized_value(self)
 |      Returns the value of the initialized variable.
 |
 |      You should use this instead of the variable itself to initialize another
 |      variable with a value that depends on the value of this variable.
 |
 |      ```python
 |      # Initialize 'v' with a random tensor.
 |      v = tf.Variable(tf.truncated_normal([10, 40]))
 |      # Use `initialized_value` to guarantee that `v` has been
 |      # initialized before its value is used to initialize `w`.
 |      # The random values are picked only once.
 |      w = tf.Variable(v.initialized_value() * 2.0)
 |      ```
 |
 |      Returns:
 |        A `Tensor` holding the value of this variable after its initializer
 |        has run.
 |
 |  ref(self)
 |      Returns a reference to this variable.
 |
 |      You usually do not need to call this method as all ops that need a reference
 |      to the variable call it automatically.
 |
 |      Returns is a `Tensor` which holds a reference to the variable.  You can
 |      assign a new value to the variable by passing the tensor to an assign op.
 |      See [`value()`](#Variable.value) if you want to get the value of the
 |      variable.
 |
 |      Returns:
 |        A `Tensor` that is a reference to the variable.
 |
 |  scatter_sub(self, sparse_delta, use_locking=False)
 |      Subtracts `IndexedSlices` from this variable.
 |
 |      This is essentially a shortcut for `scatter_sub(self, sparse_delta.indices,
 |      sparse_delta.values)`.
 |
 |      Args:
 |        sparse_delta: `IndexedSlices` to be subtracted from this variable.
 |        use_locking: If `True`, use locking during the operation.
 |
 |      Returns:
 |        A `Tensor` that will hold the new value of this variable after
 |        the scattered subtraction has completed.
 |
 |      Raises:
 |        ValueError: if `sparse_delta` is not an `IndexedSlices`.
 |
 |  to_proto(self)
 |      Converts a `Variable` to a `VariableDef` protocol buffer.
 |
 |      Returns:
 |        A `VariableDef` protocol buffer.
 |
 |  value(self)
 |      Returns the last snapshot of this variable.
 |
 |      You usually do not need to call this method as all ops that need the value
 |      of the variable call it automatically through a `convert_to_tensor()` call.
 |
 |      Returns a `Tensor` which holds the value of the variable.  You can not
 |      assign a new value to this tensor as it is not a reference to the variable.
 |      See [`ref()`](#Variable.ref) if you want to get a reference to the
 |      variable.
 |
 |      To avoid copies, if the consumer of the returned value is on the same device
 |      as the variable, this actually returns the live value of the variable, not
 |      a copy.  Updates to the variable are seen by the consumer.  If the consumer
 |      is on a different device it will get a copy of the variable.
 |
 |      Returns:
 |        A `Tensor` containing the value of the variable.
 |
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |
 |  from_proto(variable_def)
 |      Returns a `Variable` object created from `variable_def`.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables (if defined)
 |
 |  __weakref__
 |      list of weak references to the object (if defined)
 |
 |  device
 |      The device of this variable.
 |
 |  dtype
 |      The `DType` of this variable.
 |
 |  graph
 |      The `Graph` of this variable.
 |
 |  initial_value
 |      Returns the Tensor used as the initial value for the variable.
 |
 |      Note that this is different from `initialized_value()` which runs
 |      the op that initializes the variable before returning its value.
 |      This method returns the tensor that is used by the op that initializes
 |      the variable.
 |
 |      Returns:
 |        A `Tensor`.
 |
 |  initializer
 |      The initializer operation for this variable.
 |
 |  name
 |      The name of this variable.
 |
 |  op
 |      The `Operation` of this variable.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  SaveSliceInfo = <class 'tensorflow.python.ops.variables.SaveSliceInfo'...
 |      Information on how to save this Variable as a slice.
'''

# model
# y = W x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

print('y = ' + str(y))
# print('help(tf.nn.softmax) =')
# help(tf.nn.softmax)
'''
help(tf.nn.softmax) =
Help on function softmax in module tensorflow.python.ops.gen_nn_ops:

softmax(logits, name=None)
    Computes softmax activations.

    For each batch `i` and class `j` we have

        softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

    Args:
      logits: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        2-D with shape `[batch_size, num_classes]`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
'''

# To implement cross-entropy
# placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, mnist.train.labels.shape[1]])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# training setup
# learning rate = 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the variables we created
init = tf.initialize_all_tables()

# launch the model in a Session
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
