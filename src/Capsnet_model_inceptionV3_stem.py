
# coding: utf-8

# In[1]:


from keras import layers, models, optimizers
from keras.layers import Input, Conv2D, Dense
from keras.layers import Reshape, Layer, Lambda
from keras.models import Model
from keras.utils import to_categorical
from keras import initializers
from keras import backend as K

import numpy as np
import tensorflow as tf


# In[2]:


def squash(output_vector, axis=-1):
	norm = tf.reduce_sum(tf.square(output_vector), axis, keepdims=True)
	return output_vector * norm / ((1 + norm) * tf.sqrt(norm + 1.0e-10))


# In[3]:


class MaskingLayer(Layer):
	def call(self, inputs, **kwargs):
		input, mask = inputs
		return K.batch_dot(input, mask, 1)

	def compute_output_shape(self, input_shape):
		*_, output_shape = input_shape[0]
		return (None, output_shape)


# In[4]:


def PrimaryCapsule(n_vector, n_channel, n_kernel_size, n_stride, padding='valid'):
	def builder(inputs):
		output = Conv2D(filters=n_vector * n_channel, kernel_size=n_kernel_size, strides=n_stride, padding=padding)(inputs)
		output = Reshape( target_shape=[-1, n_vector], name='primary_capsule_reshape')(output)
		return Lambda(squash, name='primary_capsule_squash')(output)
	return builder


# In[5]:


class CapsuleLayer(Layer):
	def __init__(self, n_capsule, n_vec, n_routing, **kwargs):
		super(CapsuleLayer, self).__init__(**kwargs)
		self.n_capsule = n_capsule
		self.n_vector = n_vec
		self.n_routing = n_routing
		self.kernel_initializer = initializers.get('he_normal')
		self.bias_initializer = initializers.get('zeros')

	def build(self, input_shape): # input_shape is a 4D tensor
		_, self.input_n_capsule, self.input_n_vector, *_ = input_shape
		self.W = self.add_weight(shape=[self.input_n_capsule, self.n_capsule, self.input_n_vector, self.n_vector], initializer=self.kernel_initializer, name='W')
		self.bias = self.add_weight(shape=[1, self.input_n_capsule, self.n_capsule, 1, 1], initializer=self.bias_initializer, name='bias', trainable=False)
		self.built = True

	def call(self, inputs, training=None):
		input_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
		input_tiled = tf.tile(input_expand, [1, 1, self.n_capsule, 1, 1])
		input_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]), elems=input_tiled, initializer=K.zeros( [self.input_n_capsule, self.n_capsule, 1, self.n_vector]))
		for i in range(self.n_routing): # routing
			c = tf.nn.softmax(self.bias, axis=2)
			outputs = squash(tf.reduce_sum( c * input_hat, axis=1, keepdims=True))
			if i != self.n_routing - 1:
				self.bias = self.bias + tf.reduce_sum(input_hat * outputs, axis=-1, keepdims=True)
		return tf.reshape(outputs, [-1, self.n_capsule, self.n_vector])

	def compute_output_shape(self, input_shape):
		# output current layer capsules
		return (None, self.n_capsule, self.n_vector)


# In[6]:


class LengthLayer(Layer):
	def call(self, inputs, **kwargs):
		return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=False))

	def compute_output_shape(self, input_shape):
		*output_shape, _ = input_shape
		return tuple(output_shape)


# In[7]:


def margin_loss(y_ground_truth, y_prediction):
	_m_plus = 0.9
	_m_minus = 0.1
	_lambda = 0.5
	L = y_ground_truth * tf.square(tf.maximum(0., _m_plus - y_prediction)) + _lambda * ( 1 - y_ground_truth) * tf.square(tf.maximum(0., y_prediction - _m_minus))
	return tf.reduce_mean(tf.reduce_sum(L, axis=1))


# In[8]:

#inceptionv3
def conv2d_bn(x,
			  filters,
			  num_row,
			  num_col,
			  padding='same',
			  strides=(1, 1),
			  name=None):
	"""Utility function to apply conv + BN.
	# Arguments
		x: input tensor.
		filters: filters in `Conv2D`.
		num_row: height of the convolution kernel.
		num_col: width of the convolution kernel.
		padding: padding mode in `Conv2D`.
		strides: strides in `Conv2D`.
		name: name of the ops; will become `name + '_conv'`
			for the convolution and `name + '_bn'` for the
			batch norm layer.
	# Returns
		Output tensor after applying `Conv2D` and `BatchNormalization`.
	"""
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	if K.image_data_format() == 'channels_first':
		bn_axis = 1
	else:
		bn_axis = 3
	x = layers.Conv2D(
		filters, (num_row, num_col),
		strides=strides,
		padding=padding,
		use_bias=False,
		name=conv_name)(x)
	x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
	x = layers.Activation('relu', name=name)(x)
	return x

def model_sinpesos(img_width, img_height, n_class, x_all=None, y_all=None,
				   optimizer=None):
	input_shape = np.asarray([img_height, img_width, 3])
	n_routing = 1
	
	img_input = Input(shape=input_shape)

	# inceptionv3
	x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
	x = conv2d_bn(x, 32, 3, 3, padding='valid')
	x = conv2d_bn(x, 64, 3, 3)
	x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn(x, 80, 1, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, 3, padding='valid')
	x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
	
	# capsnet
	primary_capsule = PrimaryCapsule( n_vector=16, n_channel=32, n_kernel_size=9, n_stride=2)(x)

	digit_capsule = CapsuleLayer( n_capsule=n_class, n_vec=64, n_routing=n_routing, name='digit_capsule')(primary_capsule)
	output_capsule = LengthLayer(name='output_capsule')(digit_capsule)
	
	mask_input = Input(shape=(n_class, ))
	mask = MaskingLayer()([digit_capsule, mask_input])  # two inputs
	dec = Dense(512, activation='relu')(mask)
	dec = Dense(1024, activation='relu')(dec)
	# this layer depends of input shape
	dec = Dense(np.prod(input_shape), activation='sigmoid')(dec)
	dec = Reshape(input_shape)(dec)
	
	# this is the model we will train
	model = Model([img_input, mask_input], [output_capsule, dec])
	
	print(model.summary())

	return model, model

def Capsnet_sinpesos():
	return {"model": model_sinpesos, "name": "Capsnet_InceptionV3_stem_299x299", "shape": (299, 299, 3)}
