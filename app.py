#!Scripts\python
from flask import Flask, make_response, json, request, jsonify
import BBBModel as model
from random import randint
import logging

app = Flask(__name__)

model_db_path = "./db_models"
# model = BBBModel()

logging.basicConfig(filename="logs/{}.log".format(__name__), level=logging.DEBUG)
log = logging.getLogger(__name__)

##################
# demo dataset
import mxnet as mx
import numpy as np

def transform(data, label):
	return data.astype(np.float32)/126.0, label.astype(np.float32)
train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform)
num_inputs = 784
num_outputs = 10
##################



@app.route("/")
def index():
	return "Hello World!"



@app.route("/bbb/train/<int:model_id>")
def train(model_id):
	"""
	Train model with MNIST dataset
	"""	

	model_path = "{}/m{}.pkl".format(model_db_path, model_id)
	model.train(model_id, model_path, train_dataset, test_dataset, num_inputs, num_outputs,
			num_hidden_layers=2, num_hidden_units=400, 
			batch_size=128, epochs=model_id, 
			learning_rate=0.001, sigma_p=1.0)
	return make_response(jsonify({"status": "Model created and trained"}), 200)



@app.route("/bbb/predict/<int:model_id>")
def mock_predict(model_id):
	"""
	Mock the prediction of MNIST trained model with MNIST test data
	Response is a json object with two elements
		"results": [outputs]
		"labels": [labels]
	"""	

	model_path = "{}/m{}.pkl".format(model_db_path, model_id)


	##################
	# demo predict input
	for i in range(10):
		sample_idx = randint(0,len(test_dataset)-1)
		sample_test = test_dataset[sample_idx]

		if i == 0:
			sample_test_data = mx.nd.expand_dims(sample_test[0], axis = 0)	# ndarray [[data1] [data2] ...]
			sample_test_label = mx.nd.array([sample_test[1]])			# ndarray [label1 label2 ... ]
		else:
			sample_test_data = mx.nd.concat(sample_test_data, mx.nd.expand_dims(sample_test[0], axis = 0))	# ndarray [[data1] [data2] ...]
			sample_test_label = mx.nd.concat(sample_test_label, mx.nd.array([sample_test[1]]), dim = 0)				# ndarray [label1 label2 ... ]
	##################

	try: 
		output = model.predict(sample_test_data, model_path)


		# Cast each output to int
		results = []
		result_labels = []
		for i in range(output.size):
			results.append(str(mx.nd.cast(output[i], dtype='int32').asscalar()))
			result_labels.append(str(mx.nd.cast(sample_test_label[i], dtype='int32').asscalar()))
		
		response = {"results": results, "labels": result_labels}

		return make_response(jsonify(response), 200)

	except FileNotFoundError:
		response = {"error": "Model not found. Make sure you have trained the model"}
		return make_response(jsonify(response), 404)



if __name__ == '__main__':
	app.run(debug=True)
