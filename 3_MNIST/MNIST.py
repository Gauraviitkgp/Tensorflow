from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from uniform_noise import UniformNoise
from keras import backend as K
from math import pi,e,sqrt
batch_size = 128
num_classes = 10
epochs = 12
b=0.05
a=-0.05
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets #Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    # input_shape = Input((1, img_rows, img_cols))
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    # input_shape = Input((img_rows, img_cols, 1))

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')
x_train /= 255  #0-1
x_test /= 255   #0-1
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) #Convultional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Reduce dimensionality of the image
# model.add(Dropout(0.25)) #Dropout Probablity
# common_variance=((b-a)**2)/12 #Common Variance
common_entropy=(b-a)/sqrt(2*pi*e)
model.add(keras.layers.GaussianNoise(common_entropy))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(keras.layers.GaussianNoise(common_entropy))
model.add(Dense(num_classes, activation='softmax'))

# conv1   = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_shape)
# conv2   = Conv2D(64, (3, 3), activation='relu')(conv1)
# maxpool = MaxPooling2D(pool_size=(2, 2))(conv2)
# conv2n  = UniformNoise(minval=-0.05, maxval=0.05)(maxpool)
# fconv2n = Flatten()(conv2n)
# layer3  = Dense(128, activation='relu')(fconv2n)
# layer3n = UniformNoise(minval=-0.05, maxval=0.05)(layer3)
# out     = Dense(num_classes, activation='softmax')(layer3n)
# model = Model(inputs=input_shape, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])