#!Scripts\python
from flask import Flask, render_template, json, request, jsonify
import BBBtraining as bbb

from BBBModel import BBBModel

app = Flask(__name__)

model_db_path = "./db_models"
model = BBBModel()


@app.route("/")
def index():
	return "Hello World!"


@app.route("/bbb/train/mnist/<int:model_id>")
def train_mnist(model_id):
	"""
		Train model with MNIST dataset
	"""	
	model_path = "{}/m{}.pkl".format(model_db_path, model_id)
	model.train_MNIST(seed=model_id, model_path=model_path)
	return "Done"



@app.route("/bbb/predict/mock/<int:model_id>")
def mock_predict(model_id):
	"""
		Mock the prediction of MNIST trained model with MNIST test data
	"""	
	input_data = "_________???"
	model_path = "{}/m{}.pkl".format(model_db_path, model_id)
	return model.predict(input_data, model_path)

if __name__ == '__main__':
	app.run(debug=True)


