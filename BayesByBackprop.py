import logging
import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd

log = logging.getLogger(__name__)

class BayesByBackprop:

	def __init__(self, seed, num_hidden_layers, num_hidden_units, batch_size, epochs, learning_rate, sigma_p):
		log.debug("BayesByBackprop instance created with seed = {}".format(seed))

		self.config = {
			"num_hidden_layers": num_hidden_layers,
			"num_hidden_units": num_hidden_units,
			"batch_size": batch_size,
			"epochs": epochs,
			"learning_rate": learning_rate,
			"sigma_p": sigma_p,
		}


		self.ctx = mx.cpu()

		mx.random.seed(seed)
		np.random.seed(seed)

		self.num_inputs = None
		self.num_outputs = None
		self.layer_param_shapes = []
		self.mus = []
		self.rhos = []

	#######################################
	###         Model definition        ###
	#######################################

	def define_model(self, num_inputs, num_outputs):
		"""
			Define BBB model with a single input X and its label y

			Parameters
			----------
			num_inputs : int
				Number of nodes in input layer
			num_outputs : int
				Number of nodes in output layer
		"""
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs

		num_hidden = self.config['num_hidden_units']
		num_layers = self.config['num_hidden_layers']
		for i in range(num_layers + 1):
			if i == 0:
				W_shape = (num_inputs, num_hidden)
				b_shape = (num_hidden,)
			elif i == num_layers:
				W_shape = (num_hidden, num_outputs)
				b_shape = (num_outputs,)
			else:
				W_shape = (num_hidden, num_hidden)
				b_shape = (num_hidden, )
			self.layer_param_shapes.extend([W_shape, b_shape])



	def net(X, layer_params):
		"""
			Given weights and input, compute output from NN

			Parameters
			----------
			X : mxnet.ndarray
			    An input vector
			layer_params : python list of mxnet.ndarray
				Weights for each layer
		"""


		def relu(X):
			"""
				Activation function

				Parameters
				----------
				X : mxnet.ndarray
				    An input vector
			"""
			return nd.maximum(X, nd.zeros_like(X))

		layer_input = X
		for i in range(len(layer_params) // 2 - 2):
			h_linear = nd.dot(layer_input, layer_params[2*i]) + layer_params[2*i+1]
			layer_input = relu(h_linear)
		output = nd.dot(layer_input, layer_params[-2]) + layer_params[-1]	# Last layer without ReLU
		return output


	def predict(self, X):
		"""
			Given input, predict output using this model (mus)

			Parameters
			----------
			X : mxnet.ndarray
			    An input vector
		"""
		X = X.as_in_context(self.ctx).reshape((-1, self.num_inputs))
		output = BayesByBackprop.net(X, self.mus)
		predictions = nd.argmax(output, axis=1)
		return predictions



	#######################################
	###          Cost Function          ###
	#######################################


	def combined_loss(self, num_batches, params, sigmas, output, label_one_hot):
		""" 
			Total cost across batches
		"""

		def log_softmax_likelihood(yhat_linear, y):
			""" 
				Likelihood of output yhat_linear, given the label y
				yhat_linear, y: ndarray
			"""
			return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


		def log_gaussian(x, mu, sigma):
			""" 
				Log of Gaussian function log N(x|mu,sigma)
				x, mu, sigma: ndarray 
			"""
			# x, mu, sigma are of size #node_i*#node_i+1 for i-th layer
			return -0.5*np.log(2.0*np.pi) - nd.log(sigma) - 0.5 * (x-mu)**2 / (sigma ** 2)

		def gaussian_prior(x, sigma_prior):
			"""
				Gaussian Prior
				x: ndarray (weights of one layer)
			"""
			return nd.sum(log_gaussian(x, 0., sigma_prior))
		

		sigma_p = nd.array([self.config['sigma_p']], ctx=self.ctx)
		
		log_likelihood_sum = nd.sum(log_softmax_likelihood(output, label_one_hot))
		log_prior_sum = sum([nd.sum(gaussian_prior(param, sigma_p)) for param in params])
		log_var_posterior_sum = sum([nd.sum(log_gaussian(params[i], self.mus[i], sigmas[i])) for i in range(len(params))])

		return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum





	######################################
	###           Evaluation           ###
	######################################

	def evaluate_accuracy(self, data_iterator, net, layer_params):
		numerator = 0.
		denominator = 0.
		for i, (data, label) in enumerate(data_iterator):
			if i == 10:
				break
			data = data.as_in_context(self.ctx).reshape((-1, 784))
			label = label.as_in_context(self.ctx)
			predictions = self.predict(data)
			numerator += nd.sum(predictions == label)
			denominator += data.shape[0]
		return (numerator / denominator).asscalar()



	######################################
	###            Training            ###
	######################################

	def train(self, train_dataset, test_dataset):
		"""
			train_dataset: mxnet.gluon.data.dataset
						e.g. mx.gluon.data.vision.MNIST(train=True, transform=transform)

			test_dataset: mxnet.gluon.data.dataset
						e.g. mx.gluon.data.vision.MNIST(train=False, transform=transform)
		"""
		batch_size = self.config['batch_size']

		train_data = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
		test_data = mx.gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)

		num_train = sum([batch_size for i in train_data])
		num_batches = num_train / batch_size


		# 1. Init params
		weight_scale = .1
		rho_offset = -3

		mus = []
		rhos = []
		for shape in self.layer_param_shapes:
			mu = nd.random_normal(shape=shape, ctx=self.ctx, scale=weight_scale)
			rho = rho_offset + nd.zeros(shape=shape, ctx=self.ctx)
			self.mus.append(mu)
			self.rhos.append(rho)

		variational_params = self.mus + self.rhos
		for param in variational_params:
			param.attach_grad()


		# 2. Functions for main training loop
		def sample_epsilons(param_shapes):
			epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=self.ctx) for shape in param_shapes]
			return epsilons

		def transform_rhos(rhos):
			""" Apply softmax on rhos to get sigmas """
			return [nd.log(1. + nd.exp(rho)) for rho in rhos]

		def transform_gaussian_samples(mus, sigmas, epsilons):
			""" w = mu + sigma o epsilons"""
			samples = []
			for j in range(len(mus)):
				samples.append(mus[j] + sigmas[j] * epsilons[j])
			return samples

		def SGD(params, lr):
			"""
				Stochastic Gradient Descent
			"""
			for param in params:
				param[:] = param - lr * param.grad


		# 3. Complete training loop
		epochs = self.config['epochs']
		learning_rate = self.config['learning_rate']
		smoothing_constant = .01
		train_acc = []
		test_acc = []


		for e in range(epochs):
			for i, (data, label) in enumerate(train_data):
				if i == 10:
					break
				data = data.as_in_context(self.ctx).reshape((-1, self.num_inputs))
				label = label.as_in_context(self.ctx)
				label_one_hot = nd.one_hot(label, 10)

				with autograd.record():
					# sample epsilons from standard normal
					epsilons = sample_epsilons(self.layer_param_shapes)

					# compute softplus for variance
					sigmas = transform_rhos(self.rhos)

					# obtain a sample from q(w|theta) by transforming the epsilons
					# layer_params = transform_gaussian_samples(mus, sigmas, epsilons)
					layer_params = transform_gaussian_samples(self.mus, sigmas, epsilons)

					# forward-propagate the batch
					output = BayesByBackprop.net(data, layer_params)

					# calculate the loss
					loss = self.combined_loss(num_batches, layer_params, sigmas, output, label_one_hot)

				# backpropagate for gradient calculation
				loss.backward()

				# apply stochastic gradient descent to variational parameters
				SGD(variational_params, learning_rate)

				# calculate moving loss for monitoring convergence
				curr_loss = nd.mean(loss).asscalar()
				moving_loss = (curr_loss if ((i == 0) and (e == 0))
							   else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)


			test_accuracy = self.evaluate_accuracy(test_data, BayesByBackprop.net, self.mus)
			train_accuracy = self.evaluate_accuracy(train_data, BayesByBackprop.net, self.mus)
			train_acc.append(np.asscalar(train_accuracy))
			test_acc.append(np.asscalar(test_accuracy))
			print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
				  (e, moving_loss, train_accuracy, test_accuracy))

			log.info("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
				  (e, moving_loss, train_accuracy, test_accuracy))