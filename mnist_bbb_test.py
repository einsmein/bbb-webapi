import mxnet as mx
import numpy as np
from BayesByBackprop import BayesByBackprop
from BBBModel import BBBModel

################
# BayesByBackprop test
################

model = BayesByBackprop(seed=0)

def transform(data, label):
    return data.astype(np.float32)/126.0, label.astype(np.float32)

train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform)

num_inputs = 784
num_outputs = 10

# model.define_model(num_inputs, num_outputs)
# model.train(train_dataset, test_dataset)




################
# BBBModel test
################

model_db_path = "./db_models"
model_id = 0
model_path = "{}/m{}.pkl".format(model_db_path, model_id)

model = BBBModel()
# model.train_MNIST(seed=model_id, model_path=model_path)

sample_idx = 0
sample_train = train_dataset[sample_idx]
sample_train_data = sample_train[0]
sample_train_label = sample_train[1]
sample_test = test_dataset[sample_idx]
sample_test_data = sample_test[0]
sample_test_label = sample_test[1]

output = model.predict(sample_test_data, model_path)
print("label: ", sample_test_label)
print("output: ", output)