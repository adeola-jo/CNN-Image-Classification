'''
lOGGING THE TRAINING RESULTS TO W&B. 
In my case I already have an account with W&B, so I just exported my api, logged in and created a project called dl_lab2.
To log the training results to W&B, you will need to do the following:

'''

# NOTE: Edit the user and project name for logging to wandb
USERNAME = 'adeolajosepholoruntoba'
PROJECTNAME = 'dl_lab2_task3'


import torch
import torch.nn as nn
import time
import os
import math
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import wandb
import sys
import os
import random
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from torchvision.datasets import MNIST
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm   
from tabulate import tabulate
from shutil import copy as scopy


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(MNISTDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.num_examples = data.shape[0]
        assert self.num_examples == labels.shape[0]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class ConvolutionalModel(nn.Module):
    '''
    since I'm using nn.Module, I don't neeed to define weights and biases manually
    '''
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, fc2_width, input_size=28):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate output dimensions after the second pooling layer
        layers_up_to_second_pool = [self.conv1, self.pool1, self.conv2, self.pool2]
        output_dims = self._calculate_output_size(layers_up_to_second_pool, input_size)

        # Calculate flattened size for the fully connected layer
        flattened_size = self.calculate_flattened_size(output_dims)
        self.fc3 = nn.Linear(flattened_size, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, fc2_width, bias=True)

        # Initialize the parameters of the model
        self.reset_parameters()

    def forward(self, x):
        h = self.pool1(self.conv1(x)).relu()
        h = self.pool2(self.conv2(h)).relu()
        h = h.flatten(start_dim=1)
        h = self.fc3(h).relu()
        h = self.fc_logits(h)
        return h
    
    def reset_parameters(self):
        '''
        Reinitialize the parameters of the model using the He (aka Kaiming) normal initialization.
        this initialization is designed to work well with ReLU nonlinearities. It calculates the
        standard deviation of the distribution based on the number of input units, and draws
        samples from the normal distribution using that standard deviation and mean 0.
        The fan_in mode is used for the weights of the convolutional layers, and the fan_out mode
        is used for the weights of the fully connected layers.
        The formula for the standard deviation is:
            std = sqrt(2 / fan_in)
        where fan_in is the number of input units.

        The formula for the standard deviation for the fully connected layers is:
            std = sqrt(2 / fan_out)
        where fan_out is the number of output units.
        Choosing fan_in preserves the magnitude of the variance of the weights in the forward pass.
        Choosing fan_out preserves the magnitue of the variance of the gradients in the backward pass.
        The non-linearity recommended for use with this initialization is ReLU or LeakyRelu, which is the
        non-linearity used in this model.

        For more information see the documentation for nn.init.kaiming_normal_:
        https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        or the original paper:
        https://arxiv.org/pdf/1502.01852.pdfss
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def _calculate_output_size(self, layers, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.conv1.in_channels, input_size, input_size)
            for layer in layers:
                dummy_input = layer(dummy_input)
            return dummy_input.size()  # returns a tuple of (batch_size, channels, height, width)

    def calculate_flattened_size(self, output_dims):
        return torch.prod(torch.tensor(output_dims[1:])).item()


    # def get_loss(self, logits, labels):
    #     return nn.CrossEntropyLoss()(logits, labels)

    # def get_loss(self, logits, labels, epsilon=1e-15):
    #     loss = -torch.mean(torch.sum(labels * torch.log(logits + epsilon), axis=1))
    #     return loss
    

def dense_to_one_hot(y, class_count):
    return torch.eye(class_count)[y]

def _get_optimizer(config, model):
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }
    if config['optimizer'] not in optimizers:
        raise ValueError(f"Optimizer {config['optimizer']} not supported. Supported optimizers: {list(optimizers.keys())}")

    if config['use_scheduler']:
        return optimizers[config['optimizer']](model.parameters(), lr=config['lr_policy'][1]['lr'], weight_decay=config['reg_param'])
    else:
        return optimizers[config['optimizer']](model.parameters(), lr=config['lr'], weight_decay=config['reg_param'])


def draw_conv_filters(epoch, layer, name, save_dir):
    filters = layer.weight.data.cpu().numpy()
    n_filters, n_channels, filter_height, filter_width = filters.shape

    filters = (filters - filters.min()) / (filters.max() - filters.min())

    n_cols = 8
    n_rows = math.ceil(n_filters / n_cols)
    width = n_cols * filter_width + (n_cols - 1)
    height = n_rows * filter_height + (n_rows - 1)

    img = np.zeros([height, width])
    
    for i in range(n_filters):
        r = i // n_cols * (filter_height + 1)
        c = i % n_cols * (filter_width + 1)
        img[r:r+filter_height, c:c+filter_width] = np.mean(filters[i, :, :, :], axis=0)

    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_scaled = (img * 255).astype(np.uint8)

    #log the image of this particular layer, epoch and step to wandb, get the layer name from pytorch
    if wandb is not None:
        wandb.log({f'{name}_Filter_{epoch}': wandb.Image(img_scaled)})
    pass

def train(model, train_x, train_y, val_x, val_y, config, loss_fn, device):
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    model.to(device)
    optimizer = _get_optimizer(config, model)
    train_dataset = MNISTDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    if config['use_scheduler']:
        lambda_lr = lambda epoch: config['lr_policy'][epoch]['lr'] if epoch in config['lr_policy'] else optimizer.param_groups[0]['lr']
        scheduler = LambdaLR(optimizer, lr_lambda = lambda_lr)

    #use wandb to watch the model
    if wandb is not None:
        wandb.watch(model, log='all', log_freq=config['batch_size'])

    for epoch in range(1, config['max_epochs'] + 1):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config['max_epochs']}", unit='batch') as pbar:
            train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, pbar=pbar)
            layer = conv_layer_to_vis(config['conv_layer_to_vis'], model) # Get the layer to visualize
            draw_conv_filters(epoch, layer, config['conv_layer_to_vis'], config['save_dir'])
            if config['use_scheduler']:
                scheduler.step()  # Update learning rate
            evaluate("Val", val_x, val_y, model, config, loss_fn, device, epoch=epoch)

def conv_layer_to_vis(layer_name, model):
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    raise ValueError(f"Layer {layer_name} not found in model")

def calc_l2_penalty(model, reg_param):
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    return reg_param * l2_penalty

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, pbar=None):
    model.train()
    total_correct, total_loss, batch_count = 0, 0, 0
    #randomly select n batches to log images from to wandb
        # Determine the number of images to log per epoch
    num_images_to_log = config.get('num_images_to_log', 3)  # Default to 3 if not specified

    # Randomly select unique batches to log images from
    batches_to_log = random.sample(range(len(train_loader)), min(num_images_to_log, len(train_loader)))

    for batch_idx, (train_x_batch, train_y_batch) in enumerate(train_loader, 1):
        train_x_batch, train_y_batch = train_x_batch.to(device), train_y_batch.to(device)
        optimizer.zero_grad()
        logits = model(train_x_batch)
        loss = loss_fn(logits, train_y_batch) + calc_l2_penalty(model, config['reg_param'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == train_y_batch.argmax(dim=1)).sum().item()
        batch_count += train_x_batch.size(0)

        if batch_idx % 50 == 0 and pbar is not None:
            pbar.set_postfix({'Loss': total_loss / batch_count, 'Accuracy': total_correct / batch_count * 100})
            pbar.update(50)

        # Log to wandb
        if wandb is not None:
            wandb.log({'Train_epoch': epoch, 'Batch': batch_idx, 'Train_Loss': total_loss / batch_count, 'Train_accuracy': total_correct / batch_count * 100})
            # Log images to wandb
            if batch_idx in batches_to_log:
                # Log the images
                wandb.log({f'Train/image_epoch_{epoch}_batch_{batch_idx}': wandb.Image(train_x_batch[0].cpu().numpy(), caption=f'Label: {train_y_batch[0].argmax(dim=0)}')})
                
def evaluate(name, x, y, model, config, loss_fn, device, epoch=None):
    #convert the name to lowercase
    name = name.lower()
    model.eval()
    dataset = MNISTDataset(x, y)    
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
    total_correct, total_loss = 0, 0
    all_predictions = []
    all_targets = []
    num_images_to_log = config.get('num_images_to_log', 3)  # Default to 3 if not specified
    #convert the name to lowercase and chheck if it is val or valid or validation  else it is test
    if name.lower() in ['val', 'valid', 'validation', 'validation set', 'test', 'test set']:
        # Randomly select unique batches to log images from
        batches_to_log = random.sample(range(len(loader)), min(num_images_to_log, len(loader)))

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(loader, 1):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            total_correct += (logits.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
            total_loss += loss_fn(logits, y_batch).item()
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.argmax(dim=1).cpu().numpy())
            
            if wandb is not None:
                if epoch is not None and batch_idx in batches_to_log:
                    # Log the images for that epoch of validation
                    wandb.log({f'{name}/image_epoch_{epoch}_batch_{batch_idx}': wandb.Image(x_batch[0].cpu().numpy(), caption=f'Pred: {logits.argmax(dim=1)[0]}, Label: {y_batch[0].argmax(dim=0)}')})
                else:
                    # Log the images for that epoch of validation
                    wandb.log({f'{name}/image_epoch_{epoch}_batch_{batch_idx}': wandb.Image(x_batch[0].cpu().numpy(), caption=f'Pred: {logits.argmax(dim=1)[0]}, Label: {y_batch[0].argmax(dim=0)}')})

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(dataset)
    #calculate the precision, recall and f1 score over all the batches
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)
    #use tabulate to print the results to the console
    table = [['Accuracy', accuracy * 100], ['Avg Loss', avg_loss], ['Precision', precision * 100], ['Recall', recall * 100], ['F1', f1 * 100]]
    print(tabulate(table, headers=[name, 'Value'], tablefmt='outline'))
    #log to wandb
    if wandb is not None:
        wandb.log({f'{name}_accuracy': accuracy * 100, f'{name}_avg_loss': avg_loss, f'{name}_precision': precision * 100, f'{name}_recall': recall * 100, f'{name}_F1': f1 * 100})
    pass

def setup_wandb(config):
    '''
    Setup the wandb project
    '''
    try:
        # Attempt to access WandB user information
        if not wandb.api.api_key:
            raise ValueError("User not logged in")
    except (AttributeError, ValueError):
        # If user is not logged in, then login
        wandb.login(key=str(os.environ['WANDB_API_KEY']), relogin=False)
    #initialize the wandb project
    wandb.init(project=PROJECTNAME, entity=USERNAME, config=config)
    pass

def setup_paths():
    data_dir = Path(__file__).parent / 'lab2_datasets' / 'MNIST'
    save_dir = Path(__file__).parent / 'lab2_out'
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, save_dir

def setup_config(save_dir):
    return {
        'max_epochs': 8,
        'batch_size': 32,
        'save_dir': save_dir,
        'reg_param': 1e-2,
        'lr_policy': {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}},
        'optimizer': 'SGD',
        'use_scheduler': True,
        'conv_layer_to_vis': 'conv1',
        'num_images_to_log': 3,
        'save_model': True
    }

def load_data(data_dir):
    ds_train = MNIST(data_dir, train=True, download=False)
    ds_test = MNIST(data_dir, train=False)
    train_x, train_y = preprocess_data(ds_train.data, ds_train.targets)
    test_x, test_y = preprocess_data(ds_test.data, ds_test.targets)
    return train_x, train_y, test_x, test_y

def preprocess_data(data, targets):
    data = data.view(-1, 1, 28, 28).float() / 255
    mean = data.mean()
    data -= mean
    targets = dense_to_one_hot(targets, 10)
    return data, targets

def main():
    torch.manual_seed(int(time.time() * 1e6) % 2**31)
    data_dir, save_dir = setup_paths()
    config = setup_config(save_dir)

    if wandb is not None:
        setup_wandb(config)

    train_x, train_y, test_x, test_y = load_data(data_dir)
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]

    model = ConvolutionalModel(1, 16, 32, 512, 10, 28)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, train_x, train_y, valid_x, valid_y, config, criterion, device)
    evaluate("Test", test_x, test_y, model, config, criterion, device)

    model_path = os.path.join(config['save_dir'], 'model')
    torch.save(model.state_dict(), model_path)
    if wandb is not None and config['save_model'] == True:
        run_dir = wandb.run.dir 
        scopy(model_path, os.path.join(run_dir, os.path.basename(model_path)))
        wandb.save(os.path.join(run_dir, os.path.basename(model_path)), base_path=run_dir)
        # Finish the wandb run
        wandb.finish()

if __name__ == '__main__':
    main()