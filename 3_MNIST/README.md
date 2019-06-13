[Link](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

# MNIST

## [Dropouts](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
A simple and powerful regularization technique for neural networks and deep learning models is dropout.

Dropout is easily implemented by randomly selecting nodes to be dropped-out with a given probability (e.g. 20%) each weight update cycle. This is how Dropout is implemented in Keras. Dropout is only used during the training of a model and is not used when evaluating the skill of the model.

## Coding Problems
### [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/layers/MaxPooling2D)
Max pooling layer for 2D inputs (e.g. images).

* Arguments:
	* __pool_size__: An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window. Can be a single integer to specify the same value for all spatial dimensions.
	* __strides__: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation. Can be a single integer to specify the same value for all spatial dimensions.
	* __padding__: A string. The padding method, either 'valid' or 'same'. Case-insensitive.
	* __data_format__: A string. The ordering of the dimensions in the inputs. channels_last (default) and channels_first are supported. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
	* __name__: A string, the name of the layer.

* [WIKI](https://computersciencewiki.org/index.php/Max-pooling_/_Pooling):
	* Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned

	* This is done to in part to help over-fitting by providing an abstracted form of the representation. As well, it reduces the computational cost by reducing the number of parameters to learn and provides basic translation invariance to the internal representation.

	* Max pooling is done by applying a max filter to (usually) non-overlapping subregions of the initial representation.

### [Conv2D](https://keras.io/layers/convolutional/)
* Arguments

	* __filters__: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
	* __kernel_size__: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
	* __strides__: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
	* __padding__: One of "valid", "causal" or "same" (case-insensitive).  "valid" means "no padding".  "same" results in padding the input such that the output has the same length as the original input.  "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the output has the same length as the original input. Useful when modeling temporal data where the model should not violate the temporal order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
	* __data_format__: A string, one of "channels_last" (default) or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, steps, channels) (default format for temporal data in Keras) while "channels_first" corresponds to inputs with shape (batch, channels, steps).
	* __dilation_rate__: an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
	* __activation__: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation__: a(x) = x).
	* __use_bias__: Boolean, whether the layer uses a bias vector.
	* __kernel_initializer__: Initializer for the kernel weights matrix (see initializers).
	* __bias_initializer__: Initializer for the bias vector (see initializers).
	* __kernel_regularizer__: Regularizer function applied to the kernel weights matrix (see regularizer).
	* __bias_regularizer__: Regularizer function applied to the bias vector (see [regularizer](https://keras.io/regularizers/)).
	* __activity_regularizer__: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
	* __kernel_constraint__: Constraint function applied to the kernel matrix (see constraints).
	* __bias_constraint__: Constraint function applied to the bias vector (see constraints).
