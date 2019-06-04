# Tensorflow for RL

## Mathematical Meaning of Tensor
It can be looked into this [link](http://mathworld.wolfram.com/Tensor.html) 

## Initialization

### [tf Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)

```
	tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```

Inserts a placeholder for a tensor that will be always fed.

I think so it is for Input for the Neural Network

### [tf Variable](https://www.tensorflow.org/api_docs/python/tf/Variable)
A variable maintains state in the graph across calls to run(). You add a variable to the graph by constructing an instance of the class Variable.

The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. The initial value defines the type and shape of the variable. After construction, the type and shape of the variable are fixed. The value can be changed using one of the assign methods.

Just like any Tensor, variables created with Variable() can be used as inputs for other Ops in the graph. 

``` 
	# Create a variable.
	w = tf.Variable(<initial-value>, name=<optional-name>)

	# Assign a new value to the variable with `assign()` or a related method.
	w.assign(w + 1.0)
	w.assign_add(1.0)
```

#### [tf.random.uniform](https://www.tensorflow.org/api_docs/python/tf/random/uniform)
```
	tf.random.uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
	)
```
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range [minval, maxval). The lower bound minval is included in the range, while the upper bound maxval is excluded.

### [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul)
```
	tf.linalg.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
	)
```
Multiplies matrix a by matrix b, producing a * b.

Either matrix can be transposed or adjointed (conjugated and transposed) on the fly by setting one of the corresponding flag to True. These are False by default.

## Execution 

### [tf.session](https://www.tensorflow.org/api_docs/python/tf/Session)
A class for running TensorFlow operations.

A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. For example:

```
	# Build a graph.
	a = tf.constant(5.0)
	b = tf.constant(6.0)
	c = a * b

	# Launch the graph in a session.
	sess = tf.Session()

	# Evaluate the tensor `c`.
	print(sess.run(c))
```

To close a session
```
	# Using the `close()` method.
	sess = tf.Session()
	sess.run(...)
	sess.close()

	# Using the context manager.
	with tf.Session() as sess:
		sess.run(...)
```

The ConfigProto protocol buffer exposes various configuration options for a session. For example, to create a session that uses soft constraints for device placement, and log the resulting placement decisions, create a session as follows:

```
	# Launch the graph in a session that allows soft device placement and
	# logs the placement decisions.
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
	                                        log_device_placement=True)
	                                        )
``` 

Init
```
	__init__(
    target='',
    graph=None,
    config=None
	)
```

If no graph argument is specified when constructing the session, the default graph will be launched in the session

#### [Session.run](https://www.tensorflow.org/api_docs/python/tf/Session#run)
```
	run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
	)
```
Runs operations and evaluates tensors in fetches.

This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches, substituting the values in [feed_dict](https://www.aiworkbox.com/lessons/use-feed_dict-to-feed-values-to-tensorflow-placeholders) for the corresponding input values.

The fetches argument may be a single graph element, or an arbitrarily nested list, tuple, namedtuple, dict, or OrderedDict containing graph elements at its leaves.

Example
```
   a = tf.constant([10, 20])
   b = tf.constant([1.0, 2.0])
   # 'fetches' can be a singleton
   v = session.run(a)
   # v is the numpy array [10, 20]
   # 'fetches' can be a list.
   v = session.run([a, b])
   # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
   # 1-D array [1.0, 2.0]
   # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
   MyData = collections.namedtuple('MyData', ['a', 'b'])
   v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
   # v is a dict with
   # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
   # 'b' (the numpy array [1.0, 2.0])
   # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
   # [10, 20].
```

## Definations
### [tf.slice](https://www.tensorflow.org/api_docs/python/tf/slice)
Extracts a slice from a tensor.
```
tf.slice(
    input_,
    begin,
    size,
    name=None
)
```
This operation extracts a slice of size _size_ from a tensor input starting at the location specified by _begin_. The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. The starting location (begin) for the slice is represented as an offset in each dimension of input. In other words, begin[i] is the offset into the 'i'th dimension of input that you want to slice from.