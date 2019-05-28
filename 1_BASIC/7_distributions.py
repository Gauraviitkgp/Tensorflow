import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import math

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()

# =========================ESTIMATORS===================
estimator = tf.keras.estimator.model_to_estimator(model)

# ==========================MULTIPLE GPU'S==============
model.summary()

# Input Pipeline

def input_fn():
	x = np.random.random((1024, 10))
	y = np.random.randint(2, size=(1024, 1))
	x = tf.cast(x, tf.float32)
	dataset = tf.data.Dataset.from_tensor_slices((x, y))
	dataset = dataset.repeat(10)
	dataset = dataset.batch(32)
	return dataset

#Strategy
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
keras_estimator = tf.keras.estimator.model_to_estimator(
	keras_model=model,
	config=config,
	model_dir='/tmp/model_dir')

keras_estimator.train(input_fn=input_fn, steps=10)	