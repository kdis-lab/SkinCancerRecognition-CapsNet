from keras.callbacks import Callback
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, \
	average_precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
import numpy
import time
from os.path import exists

def to_label(proba):
	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')

# custom metrics
class SingleLabelAugAvg(Callback):

	def __init__(self, X_test, Y_test, x_aug, y_aug, indices):
		Callback.__init__(self)
		self.X_test = X_test
		self.Y_test = to_label(Y_test[0])
		self.x_aug = x_aug
		self.y_aug = to_label(y_aug[0])
		self.indices = indices


	def on_epoch_end(self, epoch, logs=None):
		y_pred = predict_classes(self.model, self.x_aug, self.indices)
		y_true = self.y_aug[numpy.asarray(list(self.indices.keys()))]
		
		pref = '_aug_avg'
		logs['accuracy'+pref] = accuracy_score(y_true, y_pred)
		logs['average_precision_score'+pref] = average_precision_score(y_true, y_pred)
		logs['roc_auc_score'+pref] = roc_auc_score(y_true, y_pred)
		logs['cohen_kappa_score'+pref] = cohen_kappa_score(y_true, y_pred)
		logs['matthews_corrcoef'+pref] = matthews_corrcoef(y_true, y_pred)


class BinaryAugAvg(Callback):

	def __init__(self, X_test, Y_test, x_aug, y_aug, indices):
		Callback.__init__(self)
		self.X_test = X_test
		self.Y_test = to_label(Y_test[0])
		self.x_aug = x_aug
		self.y_aug = to_label(y_aug[0])
		self.indices = indices

	def on_epoch_end(self, epoch, logs=None):
		y_pred = predict_classes(self.model, self.x_aug, self.indices)
		y_true = self.y_aug[numpy.asarray(list(self.indices.keys()))]
		
		pref = '_aug_avg'
		logs['recall_sensitivity_TPR'+pref] = recall_score(y_true, y_pred)
		logs['specificity_TNR'+pref] = recall_score(y_true, y_pred, pos_label=0)
		logs['f1_score'+pref] = f1_score(y_true, y_pred)

# copy from Sequential model
def predict_classes(model, x, indices, batch_size=None, verbose=0, steps=None):
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
	proba = proba[0]
	
	proba = numpy.asarray([numpy.average(proba[indices[i]], axis=0) for i in indices])

	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')
