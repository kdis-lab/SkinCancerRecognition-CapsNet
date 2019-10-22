# 2019 02 16

import numpy

from keras.preprocessing.image import ImageDataGenerator
import numpy
import sys
from scipy.misc import toimage
from skimage.transform import resize

import os


def resize_images(x_resize, shape):
	# before x_resize = x_resize.astype('float32') / 255
	transform = []
	for current_image in x_resize:
		transform.append(resize(current_image, shape))
	transform = numpy.asarray(transform)
	return transform

def images_augmentation(x, y, limit, imagedatagenerator=None):
	datagen = ImageDataGenerator(
# 		zca_whitening=True,
# 		featurewise_std_normalization=True,
# 		samplewise_std_normalization=True,
		horizontal_flip=True,
		vertical_flip=True,
		zoom_range=[0.65,0.85],
		rotation_range=270
	) if imagedatagenerator is None else imagedatagenerator

	datagen.fit(x)

	x_dev = []
	y_dev = []

	for x_batch, y_batch in datagen.flow(x, y, batch_size=len(x), shuffle=False):
		if len(x_dev) >= limit:
			break
		for i in range(len(x_batch)):
			if len(x_dev) >= limit:
				break
			x_dev.append(x_batch[i])
			y_dev.append(y_batch[i])

	return numpy.asarray(x_dev), numpy.asarray(y_dev)

def image_aug_balance(x,y,factor):
	# if factor = 0 , solo balancea
	# class 0 y 1

	class0 = numpy.argwhere(y==0).flatten()
	class1 = numpy.argwhere(y==1).flatten()
	may_sanos = class0.shape[0] >= class1.shape[0]

	m_x = x[class0] if may_sanos else x[class1]
	m_y = y[class0] if may_sanos else y[class1]
	m_size = m_x.shape[0]

	l_x = x[class1] if may_sanos else x[class0]
	l_y = y[class1] if may_sanos else y[class0]
	l_size = l_x.shape[0]

	if l_x.shape[0] > 0:
		x_aug, y_aug = images_augmentation(l_x,l_y, \
										   m_x.shape[0] + m_x.shape[0] * factor - l_x.shape[0])

		if x_aug.shape[0] > 0:
			l_x = numpy.append(l_x, x_aug, axis=0)
			l_y = numpy.append(l_y, y_aug, axis=0)

	if factor > 0:
		x_aug, y_aug = images_augmentation(m_x,m_y,m_x.shape[0] * factor)
		if x_aug.shape[0] > 0:
			m_x = numpy.append(m_x, x_aug, axis=0)
			m_y = numpy.append(m_y, y_aug, axis=0)

	f_x = numpy.append(m_x, l_x, axis=0)
	f_y = numpy.append(m_y, l_y, axis=0)
	
	indices = {}
	for i in range(m_size):
		indices[i] = numpy.asarray([j for j in range(i,m_x.shape[0],m_size)])
	for i in range(m_x.shape[0], m_x.shape[0] + l_size):
		indices[i] = numpy.asarray([j for j in range(i,f_x.shape[0],l_size)])
	
	return f_x, f_y, indices