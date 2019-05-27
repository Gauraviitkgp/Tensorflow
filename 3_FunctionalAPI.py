import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import math


data = np.linspace(0.0,(math.pi)/2,num=32) #Input
labels = np.sin(data) #Output

inputs = tf.keras.Input(shape=(1,)) #Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='sigmoid')(inputs) #first Layer
x = layers.Dense(64, activation='sigmoid')(x)
predictions = layers.Dense(1, activation='tanh')(x) #Final Layer

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
	loss='mse',
	metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=1, epochs=200)

result = model.predict(data, batch_size=1)

print(result)
print(labels)
