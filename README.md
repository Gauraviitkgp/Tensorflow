# Tensorflow

## How to create a virtual enviroment

[link](https://www.tensorflow.org/install/pip)

Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:

```
	virtualenv --system-site-packages -p python3 ./venv 
```

Activate the virtual environment using a shell-specific command:

```
	source ./venv/bin/activate  # sh, bash, ksh, or zsh
```

When virtualenv is active, your shell prompt is prefixed with (venv).

Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip:

```
	pip install --upgrade pip

	pip list  # show packages installed within the virtual environment
```

And to exit virtualenv later:
```
	deactivate  # don't exit until you're done using TensorFlow
```

## Tutorials 
[link](https://www.tensorflow.org/guide/keras)


### Configuring the Layers
There are many tf.keras.layers available with some common constructor parameters:

* __activation__ : Set the activation function for the layer. This parameter is specified by the name of a built-in function or as a callable object. By default, no activation is applied.

* __kernel_initializer and bias_initializer__: The initialization schemes that create the layer's weights (kernel and bias). This parameter is a name or a callable object. This defaults to the "Glorot uniform" initializer.

* __kernel_regularizer and bias_regularizer__: The regularization schemes that apply the layer's weights (kernel and bias), such as L1 or L2 regularization. By default, no regularization is applied.


### Compiling the model

[tf.keras.Model.compile](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#compile) takes three important arguments:

* __optimizer__: This object specifies the training procedure. Pass it optimizer instances from the [tf.train module](https://www.tensorflow.org/api_docs/python/tf/train), such as tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, or tf.train.GradientDescentOptimizer.

* __loss__: The function to minimize during optimization. Common choices include mean square error (mse), categorical_crossentropy, and binary_crossentropy. Loss functions are specified by name or by passing a callable object from the [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses) module.

* __metrics__: Used to monitor training. These are string names or callables from the [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) module.

### [Inputting the Data](https://www.tensorflow.org/guide/keras#input_numpy_data)

For small datasets, use in-memory NumPy arrays to train and evaluate a model. The model is "fit" to the training data using the fit method:

* __epoch__: In the neural network terminology: one epoch = one forward pass and one backward pass of all the training examples. Training is structured into epochs. An epoch is one iteration over the entire input data (this is done in smaller batches).

* __batch size__: batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need. When passed NumPy data, the model slices the data into smaller batches and iterates over these batches during training. This integer specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the batch size.

* __validation_data__: When prototyping a model, you want to easily monitor its performance on some validation data. Passing this argument—a tuple of inputs and labels—allows the model to display the loss and metrics in inference mode for the passed data, at the end of each epoch.
