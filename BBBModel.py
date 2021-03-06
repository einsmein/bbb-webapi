import logging
import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from BayesByBackprop import BayesByBackprop

import pickle



def train(seed, model_path, train_dataset, test_dataset, num_inputs, num_outputs, num_hidden_layers, num_hidden_units, batch_size, epochs, learning_rate, sigma_p):
	model = BayesByBackprop(seed, num_hidden_layers, num_hidden_units, batch_size, epochs, learning_rate, sigma_p)

	model.define_model(num_inputs, num_outputs)
	model.train(train_dataset, test_dataset)

	pickle_model(model, model_path)



def predict(input_data, model_path):
	try:
		model = unpickle_model(model_path)
		prediction = model.predict(input_data)	# mxnet ndarray of size (#input_data, output_nodes) or (output_nodes,) iff #input_data = 1
		return prediction

	except FileNotFoundError:
		print("Model has not been trained")
		raise


def pickle_model(model, path):
	""" 
	Saves the trained classifier
	"""
	with open(path, 'w+b') as f:
		pickle.dump(model, f)
		print("Pickled model at {}".format(path))


def unpickle_model(model_path):
	"""
	Load the trained classifier
	"""
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
	return model


