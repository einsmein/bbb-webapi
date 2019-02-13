import logging
import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from BayesByBackprop import BayesByBackprop

import pickle



class BBBModel:

	def __init__(self):
		"""
			Attributes
			model:	BayesByBackprop class instance
					
		"""
		# self.model = None
		logging.basicConfig(filename="logs/{}.log".format(type(self).__name__), level=logging.DEBUG)

	def train_MNIST(self, seed, model_path):

		model = BayesByBackprop(seed)

		def transform(data, label):
			return data.astype(np.float32)/126.0, label.astype(np.float32)

		train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
		test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform)

		num_inputs = 784
		num_outputs = 10

		model.define_model(num_inputs, num_outputs)
		model.train(train_dataset, test_dataset)

		self.pickle_model(model, model_path)


	def train(self, train_dataset, test_dataset, num_inputs, num_outputs, seed, model_path):
		model = BayesByBackprop(seed)

		model.define_model(num_inputs, num_outputs)
		model.train(train_dataset, test_dataset)

		self.pickle_model(model, model_path)



	def predict(self, input_data, model_path):
		"""
			input_data : mx.nd.array
		"""
		try:
			model = self.unpickle_model(model_path)
			prediction = model.predict(input_data)	# mxnet ndarray of size (#input_data, output_nodes) or (output_nodes,) iff #input_data = 1
			logging.debug("prediction: \n", prediction)
			return prediction

		except FileNotFoundError:
			return "Model has not been trained"


	def pickle_model(self, model, path):
		""" 
			Saves the trained classifier
		"""
		with open(path, 'w+b') as f:
			pickle.dump(model, f)
			print("Pickled model at {}".format(path))


	def unpickle_model(self, model_path):
		"""
			Load the trained classifier
		"""
		with open(model_path, 'rb') as f:
			model = pickle.load(f)
		return model


