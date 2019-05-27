import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import math

data = np.array([[2,3],[5,6]])#Input
labels = np.array([0,1]) #Output

class MyModel(tf.keras.Model):
	def __init__(self, num_classes=10):
		super(MyModel, self).__init__(name='my_model')
		self.num_classes=num_classes;
		#Define Layers
		self.dense_1 = layers.Dense(32, activation='sigmoid')
		self.dense_2 = layers.Dense(num_classes, activation='tanh')

	# def build(self, input_shape):
	#     shape = tf.TensorShape((input_shape[1], self.output_dim))
	#     # Create a trainable weight variable for this layer.
	#     self.kernel = self.add_weight(name='kernel',
	#                                 shape=shape,
	#                                 initializer='uniform',
	#                                 trainable=True)
	#     # Make sure to call the `build` method at the end
	#     super(MyLayer, self).build(input_shape)
	
	def call (self, inputs):
		#define your forward pass here
		x = self.dense_1(inputs)
		return self.dense_2(x)

	def compute_output_shape(self, input_shape):
		# You need to override this function if you want to use the subclassed model
	    # as part of a functional-style model.
	    # Otherwise, this method is optional.
	    shape = tf.TensorShape(input_shape).as_list()
	    shape[-1] = self.num_classes
	    return tf.TensorShape(shape)

	# def get_config(self):
	#     base_config = super(MyLayer, self).get_config()
	#     base_config['output_dim'] = self.output_dim
	#     return base_config

	# @classmethod
	# def from_config(cls, config):
	# 	return cls(**config)



model = MyModel(num_classes=1)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
	loss='mse',
	metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=1, epochs=400)

result = model.predict(data, batch_size=1)

print(result)
print(labels)
