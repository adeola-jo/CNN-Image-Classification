import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

import nn
import layers

# NOTE: EDIT THE PATH TO THE DATASET
DATA_DIR = Path(__file__).parent / 'lab2_datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'lab2_out_l2reg_0.01'

# Create DATA_DIR if it doesn't exist
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create SAVE_DIR if it doesn't exist
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-2#1e-3#1e-4
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=False), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

weight_decay = config['weight_decay']
net = []
regularizers = []

# conv1 -> regularizer1 -> pool1 -> relu1 -> conv2 -> regularizer2 -> pool2 -> relu2 -> flatten3 -> fc3 -> regularizer3 -> relu3 -> logits
# conv1: 5x5 conv, 16 filters, stride 1, pad 0
# regularizer1: L2 regularizer
# pool1: 2x2 max pool, stride 2
# relu1: relu
# conv2: 5x5 conv, 32 filters, stride 1, pad 0
# regularizer2: L2 regularizer
# pool2: 2x2 max pool, stride 2
# relu2: relu
# flatten3: flatten
# fc3: fully-connected, 512 units
# regularizer3: L2 regularizer
# relu3: relu
# logits: fully-connected, 10 units

inputs = np.random.randn(config['batch_size'], 1, 28, 28)
net += [layers.Convolution(inputs, 16, 5, "conv1")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv1_l2reg')]
net += [layers.MaxPooling(net[-1], "pool1")]
net += [layers.ReLU(net[-1], "relu1")]
net += [layers.Convolution(net[-1], 32, 5, "conv2")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv2_l2reg')]
net += [layers.MaxPooling(net[-1], "pool2")]
net += [layers.ReLU(net[-1], "relu2")]
## 7x7
net += [layers.Flatten(net[-1], "flatten3")]
net += [layers.FC(net[-1], 512, "fc3")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'fc3_l2reg')]
net += [layers.ReLU(net[-1], "relu3")]
net += [layers.FC(net[-1], 10, "logits")]

data_loss = layers.SoftmaxCrossEntropyWithLogits()
loss = layers.RegularizedLoss(data_loss, regularizers)

nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
nn.evaluate("Test", test_x, test_y, net, loss, config)
