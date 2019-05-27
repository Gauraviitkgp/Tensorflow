import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import math

# REMEMBER, size(data)==input_shape of first layer>batch_size

data = np.linspace(0.0,(math.pi)/2,num=32) #Input
labels = np.sin(data) #Output

val_data = np.linspace(0.0,(math.pi)/2,num=10)
val_labels = np.sin(val_data)


#code which makes a sequential neural network, adds 3 layers of [64,64,10]

# model = tf.keras.Sequential()

# model.add(layers.Dense(64, activation='sigmoid'))#or u can use tf.sigmoid
# model.add(layers.Dense(64, activation=tf.sigmoid))

# layers.Dense(64, kernal_regularizer=tf.keras.regularizers.l1(0.01)) # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix

# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01)) ## A linear layer with L2 regularization of factor 0.01 applied to the bias vector:

# layers.Dense(64, kernel_initializer='orthogonal') #Initializes a kernal with Random Orthogonal Matrix

# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))# A linear layer with a bias vector initialized to 2.0s:

# model.add(layers.Dense(10, activation='softmax'))


model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='sigmoid', input_shape=(1,)),
# Add another:
layers.Dense(64, activation='sigmoid'),
# Add a softmax layer with 10 output units:
layers.Dense(1, activation='tanh')])

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
# 	loss='categorical_crossentropy',
# 	metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
	loss='mse',       # mean squared error
	metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), 
	# loss=tf.keras.losses.categorical_crossentropy,
	# metrics=[tf.keras.metrics.categorical_accuracy])

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, epochs=200, batch_size=1, validation_data=(val_data, val_labels)) 

model.evaluate(data, labels, batch_size=1)

result = model.predict(data, batch_size=32)

print(result.T) #[start:stop:step]
print((labels[:]))

a=np.array([1])

print(model.predict(a,batch_size=1)) #Predict output of Sin(1)

# model.evaluate(dataset, steps=30)
