import numpy as np
# import pandas as pd
import tensorflow as tf
import sys

from skimage import io

# specify gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# allow memory growth
gpu_options = tf.GPUOptions(allow_growth=True)


def read(filename):
	val_split = 2000
	# data = np.array(pd.read_csv(filename))
	data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)
	train_y = data[:,  0]
	train_x = data[:, 1:].reshape((data.shape[0], 28, 28, 1))/255

	#############
	## verbosity
	#############
	print ('Train on {} samples, Test on {} samples.'.format(train_x.shape[0]-val_split, val_split))


	return train_x[:-val_split], train_y[:-val_split], train_x[-val_split:], train_y[-val_split:]

def RUN(train_x, train_y, valid_x, valid_y):
	##############
	## parameters
	##############
	strides = 1
	epochs  = 50
	batch_size = 128
	batch_num  = int(train_x.shape[0]/batch_size)

	images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')

	conv_1 = tf.layers.conv2d(
			inputs=images,
			filters=16,
			kernel_size=[5, 5],
			padding='same',
			activation=tf.nn.relu)
	print ('conv_1:   ', conv_1.shape)

	pool_1 = tf.layers.max_pooling2d(
			inputs=conv_1,
			pool_size=[2, 2],
			strides=2)
	print ('pool_1:   ', pool_1.shape)

	conv_2 = tf.layers.conv2d(
			inputs=pool_1,
			filters=16,
			kernel_size=[3, 3],
			padding='same',
			activation=tf.nn.relu)
	print ('conv_2:   ', conv_2.shape)

	pool_2 = tf.layers.max_pooling2d(
			inputs=conv_2,
			pool_size=[2, 2],
			strides=2,
			name='encoder_out')
	print ('pool_2:   ', pool_2.shape)

	deconv_1 = tf.layers.conv2d_transpose(
				inputs=pool_2,
				filters=16,
				kernel_size=[3, 3],
				strides=2,
				padding='same',
				activation=tf.nn.relu)
	print ('deconv_1: ', deconv_1.shape)

	deconv_2 = tf.layers.conv2d_transpose(
				inputs=deconv_1,
				filters=1,
				kernel_size=[3, 3],
				strides=2,
				padding='same',
				activation=tf.nn.relu,
				name='outputs')
	print ('deconv_2: ', deconv_2.shape)

	outputs = deconv_2

	# loss function
	tf_cost = tf.reduce_mean(tf.square(images - outputs))

	# optimizer
	tf_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(tf_cost)

	# train
	init_op = tf.global_variables_initializer()
	saver   = tf.train.Saver()

	min_loss = 1e10
	patience = 3

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		sess.run(init_op)

		for epo in range(epochs):

			p = np.random.permutation(train_x.shape[0])
			train_x = train_x[p]

			for batch in range(batch_num):

				sess.run(tf_opt, feed_dict={images: train_x[batch*batch_size:(batch+1)*batch_size]})

				if batch % 50 == 0:
					training_loss = sess.run(tf_cost, feed_dict={images: train_x[batch*batch_size:(batch+1)*batch_size]})
					val_loss      = sess.run(tf_cost, feed_dict={images: valid_x})

				print ('\rEpoch [{}/{}]   loss : {:.6f}, val_loss : {:.6f}'.format(epo+1, epochs, training_loss, val_loss), end='', flush=True)
			
			print ('')

			if min_loss > val_loss:
				save_path = saver.save(sess, './model')
				print ('		val_loss improved from {:.6f} to {:.6f}, saving model...'.format(min_loss, val_loss))
				min_loss = val_loss
				patience = 3
			else:
				print ('		val_loss did not improve')
				patience -= 1

			if patience == 0:
				print ('Early stopping')
				break


def main():
	file = 'mnist/train.csv'
	train_x, train_y, valid_x, valid_y = read(file)

	RUN(train_x, train_y, valid_x, valid_y)


	# reload
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		saver = tf.train.import_meta_graph('model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()
		images = graph.get_tensor_by_name("images:0")
		outputs = graph.get_tensor_by_name("outputs/Relu:0")
		pred = sess.run(outputs, feed_dict={images: valid_x})


	# vusualize result
	for i, p in enumerate(valid_x):
		if i > 20:
			break
		io.imsave('original/{}_org.png'.format(i), (p.reshape((28, 28))*255).astype(int))
	
	for i, p in enumerate(pred):
		if i > 20:
			break
		p = p.reshape(-1)
		p = (p - p.min()) / (p.max() - p.min())
		io.imsave('reconstruct/{}_rec.png'.format(i), (p.reshape((28, 28))*255).astype(int))




if __name__ == '__main__':
	main()