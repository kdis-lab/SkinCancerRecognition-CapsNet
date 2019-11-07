import numpy as np
import os

def check_fold(path, epochs_pre, epochs):
	# saltamos los headers
	data = np.genfromtxt(path, delimiter=',', skip_header=1,
								   dtype=np.float64)
	total = epochs + epochs_pre
	rows = data.shape[0]
	if rows < total:
		return (False,False,0,0)
	print('at least 1 fold ended')
	fold = rows//total
	if rows == fold * total:
		# true hay un fold al menos, true no hace falta truncar el fichero
		return (True, True, fold, fold * total)
	# el false es para truncar el fichero
	return (True, False, fold, fold * total)
	
def truncate_reports(path, count):
	print('truncate reports in:', count)
	with open(path, 'r') as file:
		lines = file.readlines()
		# 1 agregamos la linea de los headers
		lines = lines[:1 + count]
		lines = [i.replace('\n', '') for i in lines]
	os.remove(path)
	with open(path, 'w+') as file:
		file.write('\n'.join(lines))
