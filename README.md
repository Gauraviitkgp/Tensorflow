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


#### [If needed tf data](https://www.tensorflow.org/guide/keras#input_tfdata_datasets)

### Evaluation and Prediciton

The [tf.keras.Model.evaluate](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#evaluate) and tf.keras.Model.predict methods can use NumPy data and a tf.data.Dataset.


## Building Advanced Models

Building a model with the functional API works like this:

1) A layer instance is callable and returns a tensor.
2) Input tensors and output tensors are used to define a tf.keras.Model instance.
3) This model is trained just like the Sequential model.

### Model Subclassing
Model subclassing
Build a fully-customizable model by subclassing [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model) and defining your own forward pass. Create layers in the __init__ method and set them as attributes of the class instance. Define the forward pass in the call method.

Model subclassing is particularly useful when eager execution is enabled since the forward pass can be written imperatively.

### Adding Custom Layers
Custom layers
Create a custom layer by subclassing [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) and implementing the following methods:

* __build__: Create the weights of the layer. Add weights with the add_weight method.
* __call__: Define the forward pass.
* __compute_output_shape__: Specify how to compute the output shape of the layer given the input shape.
Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.

## Callbacks
A callback is an object passed to a model to customize and extend its behavior during training. You can write your own custom callback, or use the built-in tf.keras.callbacks that include:

* __tf.keras.callbacks.ModelCheckpoint__: Save checkpoints of your model at regular intervals.
* __tf.keras.callbacks.LearningRateScheduler__: Dynamically change the learning rate.
* __tf.keras.callbacks.EarlyStopping__: Interrupt training when validation performance has stopped improving.
* __tf.keras.callbacks.TensorBoard__: Monitor the model's behavior using TensorBoard.

## Saving And Restoring
### Saving Weights
Save and load the weights of a model using [tf.keras.Model.save_weights](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#save_weights)

By default, this saves the model's weights in the TensorFlow checkpoint file format. Weights can also be saved to the Keras HDF5 format (the default for the multi-backend implementation of Keras):

### Saving Configuration
A model's configuration can be saved—this serializes the model architecture without any weights. A saved configuration can recreate and initialize the same model, even without the code that defined the original model. Keras supports JSON and YAML serialization formats.

To load from a Json file:
```
	import json
	import pprint
	pprint.pprint(json.loads(json_string))
	fresh_model = tf.keras.models.model_from_json(json_string)
```

For YAML file:
```
	pip install PyYAML
	fresh_model = tf.keras.models.model_from_yaml(yaml_string)
```

_Caution_: Subclassed models are not serializable because their architecture is defined by the Python code in the body of the call method.

### Entire model

Entire model
The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration. This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.


