import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

import nn
import layers

DATA_DIR = Path(__file__).parent / 'lab2_datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'lab2_out'

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
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=False), MNIST(DATA_DIR, train=False, download=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x)) # by using the same mean for all datasets, we ensure that all datasets have the same scale, consistent and prevent information leak. Information leak is when information from the validation/test set is used to influence the training process. This is a common mistake in machine learning.
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))  # one-hot encoding of the labels

# net = []
# inputs = np.random.randn(config['batch_size'], 1, 28, 28)
# net += [layers.Convolution(inputs, 16, 5, "conv1")]
# net += [layers.MaxPooling(net[-1], "pool1")]
# net += [layers.ReLU(net[-1], "relu1")]
# net += [layers.Convolution(net[-1], 32, 5, "conv2")]
# net += [layers.MaxPooling(net[-1], "pool2")]
# net += [layers.ReLU(net[-1], "relu2")]
# # out = 7x7
# net += [layers.Flatten(net[-1], "flatten3")]
# net += [layers.FC(net[-1], 512, "fc3")]
# net += [layers.ReLU(net[-1], "relu3")]
# net += [layers.FC(net[-1], 10, "logits")]

# loss = layers.SoftmaxCrossEntropyWithLogits()

# nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
# nn.evaluate("Test", test_x, test_y, net, loss, config)



# network architecture
# conv for convolutional layer
# maxpooling for downsampling
# relu for non-linearity
# flatten to transform the 2D feature maps into a single long feature vector to feed into the fully-connected layer
# fc for fully-connected layer for classification
# softmax for the final output layer
# cross-entropy loss for the loss function (softmax cross-entropy with logits) suitable for multi-class classification problems

# conv1 -> pool1 -> relu1 -> conv2 -> pool2 -> relu2 -> flatten3 -> fc3 -> relu3 -> logits
# conv1: 5x5 conv, 16 filters, stride 1, pad 0
# pool1: 2x2 max pool, stride 2
# relu1: relu
# conv2: 5x5 conv, 32 filters, stride 1, pad 0
# pool2: 2x2 max pool, stride 2
# relu2: relu
# flatten3: flatten
# fc3: fully-connected, 512 units
# relu3: relu
# logits: fully-connected, 10 units

 
# net = []
# inputs = np.random.randn(config['batch_size'], 1, 28, 28)
# net += [layers.Convolution(inputs, 16, 5, "conv1")]
# net += [layers.MaxPooling(net[-1], "pool1")]
# net += [layers.ReLU(net[-1], "relu1")]
# net += [layers.Convolution(net[-1], 32, 5, "conv2")]
# net += [layers.MaxPooling(net[-1], "pool2")]
# net += [layers.ReLU(net[-1], "relu2")]
# # out = 7x7
# net += [layers.Flatten(net[-1], "flatten3")]
# net += [layers.FC(net[-1], 512, "fc3")]
# net += [layers.ReLU(net[-1], "relu3")]
# net += [layers.FC(net[-1], 10, "logits")]

# loss = layers.SoftmaxCrossEntropyWithLogits()

# nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
# nn.evaluate("Test", test_x, test_y, net, loss, config)


#NOTE: ---------------- CALCULATE THE PARAMS -----------------
net = []
inputs = np.random.randn(config['batch_size'], 1, 28, 28)
net += [layers.Convolution(inputs, 16, 5, "conv1")]
#print the shape of net
print(net[-1].shape)
print(net.shape)
import sys
sys.exit()
net += [layers.MaxPooling(net[-1], "pool1")]
print(net[-1].shape)
net += [layers.ReLU(net[-1], "relu1")]
print(net[-1].shape)
net += [layers.Convolution(net[-1], 32, 5, "conv2")]
print(net[-1].shape)
net += [layers.MaxPooling(net[-1], "pool2")]
print(net[-1].shape)
net += [layers.ReLU(net[-1], "relu2")]
print(net[-1].shape)
net += [layers.Flatten(net[-1], "flatten3")]
print(net[-1].shape)
net += [layers.FC(net[-1], 512, "fc3")]
print(net[-1].shape)
net += [layers.ReLU(net[-1], "relu3")]
print(net[-1].shape)
net += [layers.FC(net[-1], 10, "logits")]
print(net[-1].shape)

# #print the total number of parameters
total = 0
for layer in net:
    if layer.has_params:
        print(layer.name, layer.weights.shape, layer.bias.shape)
        total += layer.weights.size + layer.bias.size
print("Total number of parameters: ", total)






# # Function to calculate the shape after a convolutional layer
# def conv_output_shape(input_shape, kernel_size, padding, stride=1):
#     if padding == 'SAME':
#         return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
#     else:
#         return (input_shape[0], input_shape[1], 
#                 (input_shape[2] - kernel_size) // stride + 1, 
#                 (input_shape[3] - kernel_size) // stride + 1)

# # Function to calculate the shape after a pooling layer
# def pool_output_shape(input_shape, pool_size, stride):
#     return (input_shape[0], input_shape[1], 
#             input_shape[2] // stride, input_shape[3] // stride)

# # Initialize the network
# net_shapes = []
# input_shape = (config['batch_size'], 1, 28, 28)  # Initial input shape (N, C, H, W)

# # Adding layers and calculating shapes
# # Convolutional Layer 1
# net_shapes.append(conv_output_shape(input_shape, kernel_size=5, padding='SAME', stride=1))
# # Pooling Layer 1
# net_shapes.append(pool_output_shape(net_shapes[-1], pool_size=2, stride=2))
# # Convolutional Layer 2
# net_shapes.append(conv_output_shape(net_shapes[-1], kernel_size=5, padding='SAME', stride=1))
# # Pooling Layer 2
# net_shapes.append(pool_output_shape(net_shapes[-1], pool_size=2, stride=2))

# # Calculating the receptive field
# receptive_field = 1
# receptive_field += (5 - 1)  # First convolution layer
# receptive_field += 2 * (2 - 1)  # First pooling layer
# receptive_field += 2 * (5 - 1)  # Second convolution layer

# # Calculating memory requirements
# element_size = 4  # Assuming 4 bytes for a float
# total_memory = 0
# for shape in net_shapes:
#     total_memory += np.prod(shape) * element_size

# receptive_field, total_memory, net_shapes