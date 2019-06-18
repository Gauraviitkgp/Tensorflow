from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from uniform_noise import UniformNoise
from keras import backend as K
from math import pi,e,sqrt


class test_code(object):

	def __init__(self, UNIFORM,ERRORB,COMVAR,EPOCHS,FILENAME):
		self.epochs = EPOCHS 
		self.UNIFORM=UNIFORM
		self.ERRORB	=ERRORB
		self.COMVAR =COMVAR
		self.f=open(FILENAME,'a')
		self.f.write("No_EPOCHS:{}\tIs Uniform:{}\tError Bound:{}\tIs common variance:{}\n".format(EPOCHS,UNIFORM,ERRORB,COMVAR))

	def run(self):
		batch_size = 128
		num_classes = 10

		# input image dimensions
		img_rows, img_cols = 28, 28


		b = self.ERRORB/200
		a =-self.ERRORB/200


		# the data, split between train and test sets #Dataset
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			if self.UNIFORM == False:
				input_shape = (1, img_rows, img_cols)
			else: 
				input_shape = Input((1, img_rows, img_cols))
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			if self.UNIFORM == False:
				input_shape = (img_rows, img_cols, 1)
			else: 
				input_shape = Input((img_rows, img_cols, 1))

		x_train = x_train.astype('float32') 
		x_test = x_test.astype('float32')
		x_train /= 255  #0-1
		x_test /= 255   #0-1

		(self.f).write('x_train shape:{}\n'.format(x_train.shape))
		self.f.write('train samples:{}\n'.format(x_train.shape[0]))
		self.f.write('test samples:{}\n'.format(x_test.shape[0]))

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)


		if self.UNIFORM==False:
			common_variance=((b-a)**2)/12 #Common Variance
			common_entropy=(b-a)/sqrt(2*pi*e) #Common Entropy

			model = Sequential()
			model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))#Layer 1
			model.add(Conv2D(64, (3, 3), activation='relu')) #Layer 2
			model.add(MaxPooling2D(pool_size=(2, 2))) # Reduce dimensionality of the image
			model.add(Flatten())
			#add dense layer with sigmoid act
			model.add(Dense(128, activation='sigmoid'))
			if self.COMVAR==True:
				model.add(keras.layers.GaussianNoise(common_variance))
			else:
				model.add(keras.layers.GaussianNoise(common_entropy))
			
			model.add(Dense(num_classes, activation='softmax'))
		else:
			layer1c = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_shape)
			layer2c = Conv2D(64, (3, 3), activation='relu')(layer1c)
			mxpool2 = MaxPooling2D(pool_size=(2, 2))(layer2c)
			fmxpol2 = Flatten()(mxpool2)
			layer3  = Dense(128, activation='sigmoid')(fmxpol2)
			layer3u = UniformNoise(minval=a, maxval=b)(layer3)
			out     = Dense(num_classes, activation='softmax')(layer3u)
			model 	= Model(inputs=input_shape, outputs=out)

		model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])

		
		tensorboard=keras.callbacks.TensorBoard(log_dir='logs/{}_{}_{}'.format(self.UNIFORM,self.ERRORB,self.COMVAR),update_freq='epoch')
		model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=self.epochs,
			verbose=1,
			validation_data=(x_test, y_test),
			callbacks=[tensorboard])
		score = model.evaluate(x_test, y_test, verbose=0)
		self.f.write('Test loss:{}\n'.format(score[0]))
		self.f.write('Test accuracy:{}\n\n=========================================\n\n'.format(score[1]))
		self.f.close()