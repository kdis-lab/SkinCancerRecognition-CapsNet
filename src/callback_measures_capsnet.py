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
class SingleLabel(Callback):

	def __init__(self, X_test, Y_test):
		Callback.__init__(self)
		self.X_test = X_test
		self.Y_test = to_label(Y_test[0])

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
		self.Y_test = to_label(Y_test[0])

	def on_epoch_end(self, epoch, logs=None):
		y_pred = predict_classes(self.model, self.X_test)

		logs['recall_sensitivity_TPR'] = recall_score(self.Y_test, y_pred)
		logs['specificity_TNR'] = recall_score(self.Y_test, y_pred, pos_label=0)
		logs['f1_score'] = f1_score(self.Y_test, y_pred)
		
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
	proba = proba[0]

	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')
