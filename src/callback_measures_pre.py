from keras.callbacks import Callback
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, \
	average_precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
import numpy
import time
from os.path import exists

# custom metrics
class SingleLabel(Callback):

	def __init__(self, X_test, Y_test):
		Callback.__init__(self)
		self.X_test = X_test
		self.Y_test = Y_test

	def on_epoch_end(self, epoch, logs=None):
		y_pred = predict_classes(self.model, self.X_test)

		logs['accuracy'] = accuracy_score(self.Y_test, y_pred)
		logs['average_precision_score'] = average_precision_score(self.Y_test, y_pred)
		logs['roc_auc_score'] = roc_auc_score(self.Y_test, y_pred)
		logs['cohen_kappa_score'] = cohen_kappa_score(self.Y_test, y_pred)
		logs['matthews_corrcoef'] = matthews_corrcoef(self.Y_test, y_pred)


class Binary(Callback):

	def __init__(self, X_test, Y_test):
		Callback.__init__(self)
		self.X_test = X_test
		self.Y_test = Y_test

	def on_epoch_end(self, epoch, logs=None):
		y_pred = predict_classes(self.model, self.X_test)

		logs['recall_sensitivity_TPR'] = recall_score(self.Y_test, y_pred)
		logs['specificity_TNR'] = recall_score(self.Y_test, y_pred, pos_label=0)
		logs['f1_score'] = f1_score(self.Y_test, y_pred)


class AccumulativeTime(Callback):

	def __init__(self):
		Callback.__init__(self)
		self.start = None
		self.accumulative = 0

	def on_epoch_begin(self, epoch, logs=None):
		self.start = time.time()

	def on_epoch_end(self, epoch, logs=None):
		self.accumulative = self.accumulative + time.time() - self.start
		logs['time_accu'] = self.accumulative


class EpochsRegister(Callback):

	def __init__(self, filepath, filepathmean, epoch_start=0, do_mean=True):
		Callback.__init__(self)
		self.filepath = filepath
		self.filepathmean = filepathmean
		self.headers = True
		self.keys = []
		self.epoch_start = epoch_start
		self.do_mean = do_mean

	def on_train_begin(self, logs=None):
		if exists(self.filepath):
			file_evaluations = numpy.genfromtxt(self.filepath, delimiter=',',
												dtype=numpy.float64, names=True)
			self.keys = file_evaluations.dtype.names
			# eliminar a epoch
			self.keys = self.keys[1:]
			self.headers = False

	def on_epoch_end(self, epoch, logs=None):
		file = open(self.filepath, mode="a+")

		if self.headers:
			self.keys = logs.keys()
			self.prepare_headers(file, self.keys)
			self.headers = False

		file.write('\n')
		file.write(str(epoch + self.epoch_start) + ',')
		file.write(','.join("{}".format(logs[key]) for key in self.keys))
		file.close()

	def on_train_end(self, logs=None):
		if self.do_mean:
			self.folds_mean()

	def prepare_headers(self, file, keys):
		file.write('epoch,')
		file.write(','.join(self.keys))

	def report_dimensions(self, epochsfilepath):
		alldata = numpy.genfromtxt(epochsfilepath, delimiter=',',
								   dtype=numpy.float64, names=True)
		epochs = alldata['epoch']
		total = len(epochs)
		num_epochs = len(numpy.unique(epochs))
		return num_epochs, total // num_epochs


# copy from Sequential model
def predict_classes(model, x, batch_size=None, verbose=0, steps=None):
	"""Generate class predictions for the input samples.

	The input samples are processed batch by batch.

	# Arguments
		x: input data, as a Numpy array or list of Numpy arrays
			(if the model has multiple inputs).
		batch_size: Integer. If unspecified, it will default to 32.
		verbose: verbosity mode, 0 or 1.
		steps: Total number of steps (batches of samples)
			before declaring the prediction round finished.
			Ignored with the default value of `None`.

	# Returns
		A numpy array of class predictions.
	"""
	proba = model.predict(x, batch_size=batch_size, verbose=verbose,
						  steps=steps)

	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')
