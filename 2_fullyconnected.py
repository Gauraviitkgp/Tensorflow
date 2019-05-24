import tensorflow as tf 
from tensorflow.keras import layers

#code which makes a sequential neural network, adds 3 layers of [64,64,10]

model = tf.keras.Sequential()

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))



