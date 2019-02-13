#!Scripts\python
from flask import Flask, render_template, json, request, jsonify
import BBBtraining as bbb

from BBBModel import BBBModel

app = Flask(__name__)
model_db = {}

model_db_path = "/db_models"
model = BBBModel()

@app.route("/")
def index():
	# return render_template("index.html")
	return "Hello World!"

# @app.route("/bbb/create/<int:model_id>")
# def create_model(model_id):
# 	model_db[model_id] = None
# 	return "Done"

@app.route("/bbb/train/mnist/<int:model_id>")
def train_mnist(model_id):
	
	model_path = "{}/{}.pkl".format(model_db_path, model_id)
	model.train_MNIST(seed=model_id, model_path=model_path)
	return "Done"



@app.route("/bbb/mock_predict/<int:model_id>")
def predict(model_id):

	input_data = "_________???"
	model_path = "{}/{}.pkl".format(model_db_path, model_id)
	return predict(self, input_data, model_path)

if __name__ == '__main__':
	app.run(debug=True)


