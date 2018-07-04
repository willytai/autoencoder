import numpy as np
# import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys

# specify gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# train with 50% memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def read(filename):
	val_split = 500
	# data = np.array(pd.read_csv(filename))
	data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)
	train_y = data[:,  0]
	train_x = data[:, 1:].reshape((data.shape[0], 28, 28, 1))/255

	return train_x, train_y

def RUN(train_x, train_y):
	##############
	## parameters
	##############
	strides = 2
	epochs  = 50
	batch_size = 128
	epochs  = 20

	image = Input(shape=(28, 28, 1))

	encoder = Conv2D(32, (5, 5), padding='same', activation='relu')(image)
	encoder = MaxPooling2D((2, 2))(encoder)
	encoder = Conv2D(16, (3, 3), padding='same', activation='relu')(encoder)
	encoder = MaxPooling2D((2, 2))(encoder)

	decoder = Conv2D(16, (3, 3), padding='same', activation='relu')(encoder)
	decoder = UpSampling2D((2, 2))(decoder)
	decoder = Conv2D(1, (3, 3), padding='same', activation='relu')(decoder)
	decoder = UpSampling2D((2, 2))(decoder)

	model = Model(image, decoder)

	model.summary()

	model.compile(optimizer='adam', loss='mse')

	file_path  = 'model.h5'
	checkpoint = ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, mode='min')
	callback_list = [checkpoint, earlystop]

	model.fit(train_x, train_x, validation_split=0.05, verbose=1, epochs=epochs, batch_size=batch_size, callbacks=callback_list)




def main():
	file = 'mnist/train.csv'
	train_x, train_y = read(file)

	RUN(train_x, train_y)





if __name__ == '__main__':
	main()