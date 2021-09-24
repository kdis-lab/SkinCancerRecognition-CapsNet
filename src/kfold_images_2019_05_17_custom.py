from sklearn.model_selection import StratifiedKFold
import numpy
import keras
import json
import os
from os.path import join
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import sys
import time
from multiprocessing import Pool
from skimage.transform import resize

import sklearn
from keras.utils import to_categorical

from folds_util import check_fold, truncate_reports
from callback_measures_pre import AccumulativeTime, EpochsRegister
from callback_measures_capsnet import SingleLabel, Binary
from callback_measures_capsnet_aug import SingleLabelAug, BinaryAug
from callback_measures_capsnet_aug_avg import SingleLabelAugAvg, BinaryAugAvg
from balance import image_aug_balance
from Capsnet_model import margin_loss

def temp_file():
	tempdir = '../tempdatafold'
	if not os.path.exists(tempdir):
		os.makedirs(tempdir)
	tempfile = join(tempdir, '-'.join([str(i) for i in numpy.random.randint(1000000, size=5)] + ['.npy']))
	return tempfile

def resize_images(x_resize, shape):
	# before x_resize = x_resize.astype('float32') / 255
	transform = []
	for current_image in x_resize:
		transform.append(resize(current_image, shape))
	transform = numpy.asarray(transform)
	return transform

def remove(data, indices, batch):
	'''
	2018-09-03
	data is the labels, ex: [0,0,0,0,1,1,1,1]
	data.shape[0] >= batch
	return indices to remove of data
	indices of this data or the indices of the data in another numpy
	'''	
	array = data.astype('int64')
	nremove = array.shape[0] % batch
	counts = numpy.bincount(array)
	remove = []
	while nremove > 0:
		max = numpy.argmax(counts)
		for index in range(len(array)):
			if (array[index] == max) & (indices[index] not in remove):
				remove.append(indices[index])
				nremove = nremove - 1
				break
		counts[max] = counts[max] - 1
	return remove



def one_fold(mapitems):
	from keras import backend as k
	import tensorflow as tf
	gpu = mapitems['gpu']

	with tf.device('/gpu:' + str(gpu)):
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		try:
			k.set_session(sess)
		except:
			print('No session available')
		
		fold = mapitems['fold']
		epochs = mapitems['epochs']
		batch = mapitems['batch']
		model = mapitems['model']
		img_width = mapitems['img_width']
		img_height = mapitems['img_height']
		NUM_CLASSES = mapitems['NUM_CLASSES']
		pathX = mapitems['X']
		Y = mapitems['Y']
		pathX2 = mapitems['X2']
		Y2 = mapitems['Y2']
		optimizer = mapitems['optimizer']
		dirpath = mapitems['dirpath']
		metric = mapitems['metric']
		metric_mode = mapitems['metric_mode']
		type_class_weight = mapitems['type_class_weight']
		gpu = mapitems['gpu']

		X = numpy.load(pathX)
		os.remove(pathX)
		
		X,Y,indices = image_aug_balance(X,Y,0)
		indices = sklearn.utils.shuffle(numpy.arange(Y.shape[0]))
		X = X[indices]
		Y = Y[indices]
		
		nremove = remove(Y, [i for i in range(len(Y))], batch)
		newtrain = [i for i in range(len(Y)) if i not in nremove]
		shuffle_train = sklearn.utils.shuffle(newtrain)
		X = X[shuffle_train]
		Y = Y[shuffle_train]
		
		X2 = numpy.load(pathX2)
		os.remove(pathX2)	
		
		# test
		# callbacks
		X2_aug, Y2_aug, indices2 = image_aug_balance(X2,Y2,0)
		print('test', X2.shape, 'aug', X2_aug.shape)
		
		y_origin = numpy.copy(Y)
		Y = to_categorical(Y.astype('float32'))
		Y2 = to_categorical(Y2.astype('float32'))
		Y2_aug = to_categorical(Y2_aug.astype('float32'))
		
		train_model, base_model = model['model'](img_width, img_height, NUM_CLASSES, X, Y, optimizer)
		from keras.utils import multi_gpu_model
		parallel_model = train_model
		
		parallel_model.compile(optimizer=optimizer, loss=[ margin_loss, 'mae' ], metrics=[ margin_loss, 'mae', 'accuracy'])

		# callbacks
		print('callbacks')
		callbacks = [
			SingleLabelAugAvg([X2, Y2], [Y2, X2], [X2_aug, Y2_aug], [Y2_aug, X2_aug], indices2), BinaryAugAvg([X2, Y2], [Y2, X2], [X2_aug, Y2_aug], [Y2_aug, X2_aug], indices2),
			SingleLabelAug([X2, Y2], [Y2, X2], [X2_aug, Y2_aug], [Y2_aug, X2_aug]), BinaryAug([X2, Y2], [Y2, X2], [X2_aug, Y2_aug], [Y2_aug, X2_aug]),
			SingleLabel([X2, Y2], [Y2, X2]), Binary([X2, Y2], [Y2, X2]), AccumulativeTime()
		]
		if optimizer == "sgd":
			callbacks.append(
				ReduceLROnPlateau(monitor=metric, factor=0.2, mode=metric_mode, verbose=1))
		callbacks.append(EpochsRegister(join(dirpath, 'epochs.txt'),
										join(dirpath, 'epochs-mean.txt')))
		# end callbacks
		print('fit')											   
		parallel_model.fit([X, Y], [Y, X], epochs=epochs, batch_size=batch,
							validation_data=([X2, Y2], [Y2, X2]),
							callbacks=callbacks,
							verbose=2)


def kfold(config_file, models):
	# code
	with open(config_file) as json_data:
		configuration = json.load(json_data)

	folds = int(configuration['folds'])
	epochs = int(configuration['epochs'])
	seed = int(configuration['seed'])
	reportsDir = configuration['reportsDir']
	metric = configuration['metric']
	metric_mode = configuration['metric_mode']
	gpu = configuration['gpu']

	for dataset in configuration['datasets']:
		for batch in dataset['batch']:
			for model in models:
				for optimizer in configuration['optimizers']:
					for type_class_weight in configuration['class_weight']:
						num_batch = int(batch)
						
						if 'optimizer' in model:
							if model['optimizer'] is not None:
								final_optimizer = model['optimizer']
							else:
								final_optimizer = optimizer
						else:
							final_optimizer = optimizer
						
						dirpath = join(reportsDir, dataset['name'], model['name'], "batch_" + str(batch),
									   final_optimizer, type_class_weight)

						try:
							# if this experiment was finished continue
							if os.path.exists(join(dirpath, 'summary.txt')):
								continue
							fold_ended = 0
							if os.path.exists(join(dirpath, 'epochs.txt')):
								info = check_fold(join(dirpath, 'epochs.txt'), 0, epochs)
								
								if info[0]:
									fold_ended = info[2]
									if not info[1]:
										truncate_reports(join(dirpath, 'epochs.txt'), info[3])
								else:
									os.remove(join(dirpath, 'epochs.txt'))
									if os.path.exists(join(dirpath, 'epochs-mean.txt')):
										os.remove(join(dirpath, 'epochs-mean.txt'))

							if not os.path.exists(dirpath):
								os.makedirs(dirpath)

							# fix random seed for reproducibility
							numpy.random.seed(seed)

							X = numpy.load(dataset['x'])
							X = X.astype('float32') / 255
							img_width = len(X[0][0])
							img_height = len(X[0])
							# transform image according with model
							if 'shape' in model:
								model_shape = model['shape']
								if img_height != model_shape[0] | img_width != model_shape[1]:
									print('>> resize images', dirpath)
									X = resize_images(X, model_shape)
									img_height = model_shape[0]
									img_width = model_shape[1]
							
							Y = numpy.load(dataset['y'])

							NUM_CLASSES = keras.utils.to_categorical(Y).shape[1]

							# define N-fold cross validation test harness
							kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

							fold = 0
							for train, test in kfold.split(X, Y):
								if fold < fold_ended:
									print('jumping', fold, 'fold')
									fold = fold + 1
									continue
									
								new_x_train = X[train]								
								file_x_train = temp_file()
								numpy.save(file_x_train, new_x_train)
								del new_x_train
								
								new_x_test = X[test]
								file_x_test = temp_file()
								numpy.save(file_x_test, new_x_test)
								del new_x_test
								
								time.sleep(10)
								start_time = time.time()
								with Pool(processes=1) as pool:
									pool.map(one_fold, [{
										'fold': fold,
										'epochs': epochs,
										'batch': num_batch,
										'model': model,
										'img_width': img_width,
										'img_height': img_height,
										'NUM_CLASSES': NUM_CLASSES,
										'X': file_x_train,
										'Y': Y[train],
										'X2': file_x_test,
										'Y2': Y[test],
										'optimizer': final_optimizer,
										'dirpath': dirpath,
										'metric': metric,
										'metric_mode': metric_mode,
										'type_class_weight': type_class_weight,
										'gpu': gpu
									}])
									print('>> fold', dirpath)
									print('>> fold', fold, 'completed in', str(time.time() - start_time), 'seconds')
									fold = fold + 1


							# delete dataset
							del X
							del Y
							time.sleep(3)
						except Exception as exception:
							print('error >>', dirpath)
							print('reason >>', exception)



if __name__ == '__main__':
	config_file = str(sys.argv[1])
	from Capsnet_model_inceptionV3 import Capsnet_sinpesos as CI
	from Capsnet_model_inceptionV3_stem import Capsnet_sinpesos as CS
	from Capsnet_model_inceptionV3_stem import Capsnet_pretrained as CSP
	
	kfold(config_file, [
		# 299x299
		# clasico
		CI(),
		CS(),
		CSP()
	])
