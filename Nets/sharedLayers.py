import tensorflow as tf
import os

from Data_utils import preprocessing


INITIALIZER_CONV = tf.contrib.layers.xavier_initializer()
INITIALIZER_BIAS = tf.constant_initializer(0.0)

INITIALIZER_ZEROS = tf.constant_initializer(0.0)
INITIALIZER_ONES = tf.constant_initializer(1.0)

def correlation(x,y,max_disp, name='corr', stride=1):
	with tf.variable_scope(name):
		corr_tensors = []
		y_shape = tf.shape(y)
		y_feature = tf.pad(y,[[0,0],[0,0],[max_disp,max_disp],[0,0]])
		for i in range(-max_disp, max_disp+1,stride):
			shifted = tf.slice(y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1])
			corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))

		result = tf.concat(corr_tensors,axis=-1)
		return result

def get_variable(name,kernel_shape,initializer,variable_collection,trainable=True):
	if variable_collection is None:
		return tf.get_variable(name,kernel_shape,initializer=initializer,trainable=trainable)
	else:
		return variable_collection[tf.get_variable_scope().name+'/'+name]

def batch_norm(x, training=False, momentum=0.99, variable_collection=None):
	with tf.variable_scope('bn'):
		n_out = x.get_shape()[-1].value
		beta = get_variable('beta', [n_out],INITIALIZER_ZEROS,variable_collection)
		gamma = get_variable('gamma', [n_out],INITIALIZER_ONES,variable_collection)
		#compute moments of incoming batch
		#axes = [0,1,2] if len(x.get_shape())==4 else [0,1,2,3]
		axes = list(range(len(x.get_shape())-1))
		batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
		#create explicit variable for keeping mean and variance without tensorflow bullshit
		mean = get_variable('moving_mean',batch_mean.get_shape(),INITIALIZER_ZEROS,variable_collection,trainable=False)
		var = get_variable('moving_variance', batch_var.get_shape(),INITIALIZER_ONES,variable_collection,trainable=False)
		if training:
			#before applying any training step be shure to increment the moving average and update mean and var
			update_average_mean = mean.assign(momentum*mean+(1-momentum)*batch_mean)
			update_average_var = var.assign(momentum*var+(1-momentum)*batch_var)
			with tf.control_dependencies([update_average_mean,update_average_var]):
				mean_eval = tf.identity(batch_mean)
				var_eval = tf.identity(batch_var)
		else:
			mean_eval = mean
			var_eval = var

		#finally perform batch norm
		normed = tf.nn.batch_normalization(x, mean_eval, var_eval, beta, gamma, 1e-3)
		return normed

def conv2d(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', bn=False, training=False, variable_collection=None):
	assert(len(kernel_shape)==4)
	with tf.variable_scope(name, reuse=reuse):
		W = get_variable(wName, kernel_shape, INITIALIZER_CONV, variable_collection)
		b = get_variable(bName, kernel_shape[3], INITIALIZER_BIAS, variable_collection)
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
		x = tf.nn.bias_add(x, b)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x,training=training_bn,momentum=0.99,variable_collection=variable_collection)
		x = activation(x)
		return x

def conv3d(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', bn=False, training=False, variable_collection=None):
	assert(len(kernel_shape)==5)
	with tf.variable_scope(name,reuse=reuse):
		W = get_variable(wName, kernel_shape, INITIALIZER_CONV, variable_collection)
		b = get_variable(bName, kernel_shape[4], INITIALIZER_BIAS, variable_collection)
		x = tf.nn.conv3d(x, W, strides=[1,strides,strides,strides,1],padding=padding)
		x = tf.nn.bias_add(x, b)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x, training=training_bn, momentum=0.99, variable_collection=variable_collection)
		x = activation(x)
		return x

def dilated_conv2d(x, kernel_shape, rate=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='dilated_conv', reuse=False, wName='weights', bName='biases',  bn=False, training=False, variable_collection=None):
	with tf.variable_scope(name, reuse=reuse):
		weights = get_variable(wName, kernel_shape, INITIALIZER_CONV, variable_collection)
		biases = get_variable(bName, kernel_shape[3], INITIALIZER_BIAS, variable_collection)
		x = tf.nn.atrous_conv2d(x, weights, rate=rate, padding=padding)
		x = tf.nn.bias_add(x, biases)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x,training=training_bn,momentum=0.99,variable_collection=variable_collection)
		x = activation(x)
		return x


def conv2d_transpose(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), name='conv', reuse=False, wName='weights', bName='bias',  bn=False, training=False, variable_collection=None):
	with tf.variable_scope(name, reuse=reuse):
		W = get_variable(wName, kernel_shape,INITIALIZER_CONV, variable_collection)
		tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
		b = get_variable(bName, kernel_shape[2], INITIALIZER_BIAS, variable_collection)
		x_shape = tf.shape(x)
		output_shape = [x_shape[0], x_shape[1] * strides,x_shape[2] * strides, kernel_shape[2]]
		x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x,training=training_bn,momentum=0.99,variable_collection=variable_collection)
		x = activation(x)
		return x

def conv3d_transpose(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), name='conv', reuse=False, wName='weights', bName='bias',  bn=False, training=False, variable_collection=None):
	with tf.variable_scope(name, reuse=reuse):
		W = get_variable(wName, kernel_shape,INITIALIZER_CONV, variable_collection)
		tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
		b = get_variable(bName, kernel_shape[3], INITIALIZER_BIAS, variable_collection)
		x_shape = tf.shape(x)
		output_shape = [x_shape[0], x_shape[1] * strides,x_shape[2] * strides, x_shape[3]*strides, kernel_shape[3]]
		x = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, strides, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x,training=training_bn,momentum=0.99,variable_collection=variable_collection)
		x = activation(x)
		return x

def depthwise_conv(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', bn=False, training=False, variable_collection=None):
	with tf.variable_scope(name, reuse=reuse):
		w = get_variable(wName, kernel_shape, INITIALIZER_CONV, variable_collection)
		b = get_variable(bName, kernel_shape[3]*kernel_shape[2], INITIALIZER_BIAS, variable_collection)
		x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
		x = tf.nn.bias_add(x, b)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x, training=training_bn,momentum=0.9,variable_collection=variable_collection9)
		x = activation(x)
		return x

def separable_conv2d(x, kernel_shape, channel_multiplier=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False, variable_collection=None):
	with tf.variable_scope(name, reuse=reuse):
		#detpthwise conv
		depthwise_conv_kernel = [kernel_shape[0],kernel_shape[1],kernel_shape[2],channel_multiplier]
		x = depthwise_conv(x,depthwise_conv_kernel,strides=strides,activation=lambda x: tf.maximum(0.1 * x, x),padding=padding,name='depthwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training, variable_collection=variable_collection)

		#pointwise_conv
		pointwise_conv_kernel = [1,1,x.get_shape()[-1].value,kernel_shape[-1]]
		x = conv2d(x,pointwise_conv_kernel,strides=strides,activation=activation,padding=padding,name='pointwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training, variable_collection=variable_collection)

		return x

def grouped_conv2d(x, kernel_shape, num_groups=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False, variable_collection=None):
	with tf.variable_scope(name,reuse=reuse):
		w = get_variable(wName, kernel_shape,INITIALIZER_CONV, variable_collection)
		b = get_variable(bName, kernel_shape[3], INITIALIZER_BIAS, variable_collection)

		input_groups = tf.split(x,num_or_size_splits=num_groups,axis=-1)
		kernel_groups = tf.split(w, num_or_size_splits=num_groups, axis=2)
		bias_group = tf.split(b,num_or_size_splits=num_groups,axis=-1)
		output_groups = [tf.nn.conv2d(i, k,[1,strides,strides,1],padding=padding)+bb for i, k,bb in zip(input_groups, kernel_groups,bias_group)]
		# Concatenate the groups
		x = tf.concat(output_groups,axis=3)
		if bn:
			training_bn = training and (not reuse)
			x = batch_norm(x,training=training_bn,momentum=0.99,variable_collection=variable_collection)
		x = activation(x)  
		return x

def channel_shuffle_inside_group(x, num_groups, name='shuffle'):
	with tf.variable_scope(name):
		_, h, w, c = x.shape.as_list()
		x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
		x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
		output = tf.reshape(x_transposed, [-1, h, w, c])
	return output


def normalize(input_data, blind=False):
	if blind:
		#normalize as performing a mean over all prediction
		normalizer = tf.cast(tf.size(input_data), tf.float32)
	else:
		#normalize such that the weights sum up to 1
		is_all_zeros = tf.equal(tf.count_nonzero(input_data), 0)
		normalizer = tf.reduce_sum(input_data) + tf.cast(is_all_zeros, tf.float32)
	return input_data/normalizer

def weighting_network(input_data,reuse=False, kernel_size=3,kernel_channels=64,training=False,activation=lambda x:tf.nn.sigmoid(x), scale_factor=4):
	num_channel = input_data.shape[-1].value
	full_res_shape = input_data.get_shape().as_list()
	input_data = tf.stop_gradient(input_data)
	with tf.variable_scope('feed_forward',reuse=reuse): 
		input_data = preprocessing.rescale_image(input_data,[x//scale_factor for x in full_res_shape[1:3]])
		x = conv2d(input_data,[kernel_size,kernel_size,num_channel,kernel_channels/2],training=training, padding="SAME",name="conv1",bn=True)
		x = conv2d(x,[kernel_size,kernel_size,kernel_channels/2,kernel_channels],training=training, padding="SAME",name="conv2",bn=True)
		weight = conv2d(x, [kernel_size,kernel_size,kernel_channels,1],activation=activation,training=True, padding="SAME",name="conv3")
		weight_scaled = preprocessing.rescale_image(weight, full_res_shape[1:3])
		#normalize weights
		weight_scaled = normalize(weight_scaled, blind=True)

	return weight_scaled, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=tf.get_variable_scope().name+'/feed_forward')


