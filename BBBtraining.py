
# coding: utf-8


from __future__ import print_function
import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from matplotlib import pyplot as plt

config = {
    "num_hidden_layers": 2,
    "num_hidden_units": 400,
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_samples": 1,
    "pi": 0.25,
    "sigma_p": 1.0,
    "sigma_p1": 0.75,
    "sigma_p2": 0.1,
}


ctx = mx.cpu()
mx.random.seed(0)
np.random.seed(0)


# Loading training and testing datasets

def transform(data, label):
    return data.astype(np.float32)/126.0, label.astype(np.float32)

# mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = config['batch_size']


train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                     batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                    batch_size, shuffle=False)

num_train = sum([batch_size for i in train_data])
num_batches = num_train / batch_size



#######################################
###         Model definition        ###
#######################################

# Activation function
def relu(X):
    return nd.maximum(X, nd.zeros_like(X))

# Neural net modeling
num_layers = config['num_hidden_layers']
    
# Shape of params in each layer
# [w1 b1 w2 b2 ...]
layer_param_shapes = []
num_hidden = config['num_hidden_units']
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
    layer_param_shapes.extend([W_shape, b_shape])
print(layer_param_shapes)

# Compute output of NN
# dot correct???
def net(X, layer_params):
    layer_input = X
    for i in range(len(layer_params) // 2 - 2):
        h_linear = nd.dot(layer_input, layer_params[2*i]) + layer_params[2*i+1] # Wt * X + b
        layer_input = relu(h_linear)
    # Last layer without ReLU
    output = nd.dot(layer_input, layer_params[-2]) + layer_params[-1]
    return output


#######################################
###          Cost Function          ###
#######################################


# Likelihood
def log_softmax_likelihood(yhat_linear, y):
    return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

# Prior (Gaussian)
def log_gaussian(x, mu, sigma):
    # x, mu, sigma are of size #node_i*#node_i+1 for i-th layer
    return -0.5*np.log(2.0*np.pi) - nd.log(sigma) - 0.5 * (x-mu)**2 / (sigma ** 2)

def gaussian_prior(x):
    # layer_params = [... w_i ...] --> x = w_i 
    sigma_p = nd.array([config['sigma_p']], ctx=ctx)
    return nd.sum(log_gaussian(x, 0., sigma_p))

# Variation posterior
def log_posterior(x, mu, sigma):
    return log_gaussian(x, mu, sigma)

# Total cost across batches
def combined_loss(output, label_one_hot, params, mus, sigmas, log_prior, log_likelihood):
    
    log_likelihood_sum = nd.sum(log_likelihood(output, label_one_hot))
    
    log_prior_sum = sum([nd.sum(log_prior(param)) for param in params])
    
    log_var_posterior_sum = sum([nd.sum(log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])

    return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum


######################################
###          Optimization          ###
######################################

# Stochastic Gradient Descent
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


######################################
###           Evaluation           ###
######################################

def evaluate_accuracy(data_iterator, net, layer_params):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data, layer_params)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()







######################################
###         Main Execution         ###
######################################

def train():
    # 1. Init params
    weight_scale = .1
    rho_offset = -3

    # initialize variational parameters; mean and variance for each weight
    mus = []
    rhos = []

    for shape in layer_param_shapes:
        mu = nd.random_normal(shape=shape, ctx=ctx, scale=weight_scale)
        rho = rho_offset + nd.zeros(shape=shape, ctx=ctx)
        mus.append(mu)
        rhos.append(rho)

    variational_params = mus + rhos

    for param in variational_params:
        param.attach_grad()


    # 2. Functions for main training loop
    def sample_epsilons(param_shapes):
        epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
        return epsilons

    def softplus(x):
        return nd.log(1. + nd.exp(x))

    def transform_rhos(rhos):
        return [softplus(rho) for rho in rhos]

    def transform_gaussian_samples(mus, sigmas, epsilons):
        samples = []
        for j in range(len(mus)):
            samples.append(mus[j] + sigmas[j] * epsilons[j])
        return samples

    # 3. Complete training loop
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    smoothing_constant = .01
    train_acc = []
    test_acc = []

    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)

            with autograd.record():
                # sample epsilons from standard normal
                epsilons = sample_epsilons(layer_param_shapes)

                # compute softplus for variance
                sigmas = transform_rhos(rhos)

                # obtain a sample from q(w|theta) by transforming the epsilons
                layer_params = transform_gaussian_samples(mus, sigmas, epsilons)

                # forward-propagate the batch
                output = net(data, layer_params)

                # calculate the loss
                loss = combined_loss(output, label_one_hot, layer_params, mus, sigmas, gaussian_prior, log_softmax_likelihood)

            # backpropagate for gradient calculation
            loss.backward()

            # apply stochastic gradient descent to variational parameters
            SGD(variational_params, learning_rate)

            # calculate moving loss for monitoring convergence
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)


    test_accuracy = evaluate_accuracy(test_data, net, mus)
    train_accuracy = evaluate_accuracy(train_data, net, mus)
    train_acc.append(np.asscalar(train_accuracy))
    test_acc.append(np.asscalar(test_accuracy))
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))

    return [mu.asnumpy().tolist() for mu in mus] 


    # plt.plot(train_acc)
    # plt.plot(test_acc)
    # plt.show()

