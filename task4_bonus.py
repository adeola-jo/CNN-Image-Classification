import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
import logging
import wandb
import math
import time
import random
import shutil
import skimage as ski
import io
from copy import deepcopy
from shutil import copy as scopy
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, Subset
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from seaborn import heatmap
from tabulate import tabulate
from torchsummary import summary    


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
WANDB_USERNAME = 'adeolajosepholoruntoba'

class ConfigurationError(Exception):
    """Exception raised for errors in the configuration file."""
    pass

class ModelConstructionError(Exception):
    """Exception raised for errors during model construction."""
    pass

class DotDict(dict):
    """Dictionary subclass that allows dot notation access to its elements and all sub-dictionaries."""
    
    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [DotDict(item) if isinstance(item, dict) else item for item in value]
            self[key] = value
    
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getitem__(self, item):
        value = super(DotDict, self).__getitem__(item)
        return DotDict(value) if isinstance(value, dict) else value
    
    def __deepcopy__(self, memo):
        return DotDict(deepcopy(dict(self), memo))

class CIFARDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(CIFARDataLoader, self).__init__()
        self.data = data
        self.labels = labels
        self.num_examples = data.shape[0]
        assert self.num_examples == labels.shape[0]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MulticlassHingeLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super(MulticlassHingeLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Compute the multiclass hinge loss between `logits` and the ground truth `target`.
        Args:
            logits (torch.Tensor): Tensor of logits of shape (B, C).
            target (torch.Tensor): Ground truth labels, tensor of shape (B,).
        Returns:
            torch.Tensor: Scalar tensor containing the loss.
        """

        num_classes = logits.size(1)
        
        correct_class_mask = torch.nn.functional.one_hot(target, num_classes).bool()
        correct_class_logits = torch.masked_select(logits, correct_class_mask)
        incorrect_class_logits = torch.masked_select(logits, ~correct_class_mask)

        correct_class_logits = correct_class_logits.view(-1, 1)
        incorrect_class_logits = incorrect_class_logits.view(-1, num_classes - 1)

        differences = incorrect_class_logits - correct_class_logits + self.delta
        hinge_loss = torch.max(differences, torch.tensor(0.0)).sum(dim=1)

        if self.reduction == 'mean':
            return hinge_loss.mean()
        elif self.reduction == 'sum':
            return hinge_loss.sum()
        elif self.reduction == 'none':
            return hinge_loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")
        
    # def backward(self):
    #     pass

class DynamicNet(nn.Module):
    '''
    A class that allows you to build a neural network dynamically from a configuration file
    the configuration file is a YAML file that contains the architecture of the neural network 
    along with the hyperparameters
    '''

    def __init__(self, config, verbose=False, device='cpu'):
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleDict()
        #move self.layers to the device
        # self.layers.to(device)
        self.device = device
        self.input_shape = config.input_shape
        self.config = config
        self.verbose = config.verbose
        self.device = device
        self.verbose2 = verbose # quick fix not to print the model architecture any other time aside from the first time
        #load the configuration and use it to build the layers
        self._build_layers(config.architecture)
        
        #reset the parameters of the model
        self.reset_parameters()

        # Print the model architecture
        if self.verbose and self.verbose2:
            logger.info(f"Successfully Created model with architecture")
            # print("---------------------------------------------")
            # print(self.__repr__())
            # print("---------------------------------------------")
            summary(self, input_size=[self.input_shape])


    def _build_layers(self, architecture):
        architecture = deepcopy(architecture)
        input_shape = self.input_shape
        dummy_input = torch.rand((1, *input_shape))
        current_channels = input_shape[0]  # Track the current number of channels

        for layer_name, layer_info in architecture.items():
            layer_type = layer_info.pop('type')
            if self.verbose and self.verbose2:
                logger.info(f"Added layer {layer_name} with config: {layer_info}")

            # Set 'in_channels' for Conv2d layers dynamically based on the current channel size
            if layer_type == 'Conv2d':
                if 'in_channels' not in layer_info:
                    layer_info['in_channels'] = current_channels
                current_channels = layer_info['out_channels']

            # Handle Linear layer 'in_features'
            if layer_type == 'Linear' and 'in_features' not in layer_info:
                if dummy_input.dim() != 2:
                    dummy_input = dummy_input.view(dummy_input.size(0), -1)
                layer_info['in_features'] = dummy_input.size(-1)
            
            if layer_type == 'BatchNorm2d':
                if 'num_features' not in layer_info:
                    layer_info['num_features'] = current_channels

            if layer_type == 'Dropout': 
                if 'p' not in layer_info:
                    layer_info['p'] = 0.5

            if layer_type == 'MaxPool2d':
                if 'kernel_size' not in layer_info:
                    layer_info['kernel_size'] = 2
                if 'stride' not in layer_info:
                    layer_info['stride'] = 2
                if 'padding' not in layer_info:
                    layer_info['padding'] = 0

            if layer_type == 'AvgPool2d':
                if 'kernel_size' not in layer_info:
                    layer_info['kernel_size'] = 2
                if 'stride' not in layer_info:
                    layer_info['stride'] = 2
                if 'padding' not in layer_info:
                    layer_info['padding'] = 0


            layer = getattr(nn, layer_type)(**layer_info)
            dummy_input = layer(dummy_input)

            # Update current_channels if the layer changes the channel dimension
            if hasattr(layer, 'out_channels'):
                current_channels = layer.out_channels

            self.layers[layer_name] = layer
          
        #check if the shape of the final output matches the number of classes
        assert dummy_input.shape[1] == self.config.num_classes, f"Final output shape {dummy_input.shape} does not match the number of classes {self.config.num_classes}"
        if self.verbose and self.verbose2:  
            logger.info(f"Model construction complete. Final output shape: {dummy_input.shape}")
        
    def forward(self, x):
        self.layers.to(self.device)
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    def stable_softmax(self, x):
        '''
        A stable implementation of the softmax function
        '''
        exps = torch.exp(x - torch.max(x))
        return exps / torch.sum(exps)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            #ensure its not the last layer use negative indexing to get the last layer out of the self.layers dictionary. Remember there is a -1 index in python
            elif isinstance(m, nn.Linear) and m is not self.layers[list(self.layers.keys())[-1]]:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        # Reset the parameters of the last layer using .reset_parameters()
        self.layers[list(self.layers.keys())[-1]].reset_parameters()
    
    def __repr__(self):
        return f"DynamicNet({self.layers})"
    

def preprocess_dataset(data, targets):
    # Define a transform to convert the data from uint8 to float32 and normalize it to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] and changes shape to [C, H, W]
    ])

    # Apply the transform to each data point
    processed_data = torch.stack([transform(d) for d in data])
    # Convert the targets to a tensor
    targets = torch.tensor(targets, dtype=torch.long)
    return processed_data, targets

def plot_training_progress(data, show=False, save_dir=None):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    if save_dir is not None:
        save_path = os.path.join(save_dir, 'training_plot.png')
        print('Plotting in: ', save_path)
        plt.savefig(save_path)
    if show:
        plt.show()
    if wandb is not None:
        #save the image in the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        #log the image to wandb
        wandb.log({'Metric plots': wandb.Image(image)})
        buf.close()

def setup_wandb(config):
    '''
    Setup the wandb project
    '''
    try:
        # Attempt to access WandB user information
        if not wandb.api.api_key:
            raise ValueError("User not logged in")
    except (AttributeError, ValueError):
        # try to read the api key from the config file. If its empty then read it from the environment variable
        if config.WANDB_API_KEY != '':
            wandb.login(key=config.WANDB_API_KEY, relogin=False)
        else:
            wandb.login(key=str(os.environ['WANDB_API_KEY']), relogin=False)
        #flag error if the api key is not found in the config file or the environment variable
        if not wandb.api.api_key:
            raise ValueError("User not logged in")
    #initialize the wandb project
    wandb.init(project=config.PROJECTNAME, entity=config.ENTITY, config=config)


def draw_conv_filters(epoch, step, layer, name, save_dir):
    #depending on the type of device we are using, we need to move 
    #the layer to the cpu before we can access its weight data

    w = layer.weight.data.cpu().numpy() if torch.cuda.is_available() else layer.weight.data.numpy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = f'{name}_epoch_%02d_step_%06d.png' % (epoch, step)
    #log the image of this particular layer, epoch and step to wandb, get the layer name from pytorch
    if wandb is not None:
        wandb.log({f'{filename}': wandb.Image(img)})
    pass

def conv_layer_to_vis(layer_name, model):
    for name, layer in model.named_modules():
        if name.split('.')[-1] == layer_name:
            return layer
    # raise ValueError(f"Layer {layer_name} not found in model")
    return None

def calc_l2_penalty(model, reg_param):
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(param ** 2)
    return reg_param * l2_penalty

def get_optimizer(config, model):
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
    }
    optimizer_type = config.hyperparameters.optimizer.type
    optimizer_params = config.hyperparameters.optimizer.params
    #print the optimizer parameters
    if config.verbose:
        logger.info(f"Using optimizer {optimizer_type} with params {optimizer_params}")

    if optimizer_type not in optimizers:
        raise ValueError(f"Optimizer {optimizer_type} not supported. Supported optimizers: {list(optimizers.keys())}")

    optimizer = optimizers[optimizer_type](model.parameters(), **optimizer_params)
    return optimizer

def get_loss_function(config):
    if str(config.hyperparameters.loss_function) in config.custom_loss_functions.keys():
        return deepcopy(args.custom_loss_functions[config.hyperparameters.loss_function])
    else:
        return getattr(nn, config.hyperparameters.loss_function)()
 
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def get_lr_adjuster(config, optimizer):
    if 'lr_policy' in config.hyperparameters:
        lr_policy = config.hyperparameters.lr_policy
        last_lr = config.hyperparameters.optimizer.params.lr
        sorted_epochs = sorted(lr_policy.keys())

        def lr_lambda(epoch):
            nonlocal last_lr
            new_lr = None
            # Offset the epoch by 1 to align with your training loop's counting
            adjusted_epoch = epoch + 1
            for start_epoch in sorted_epochs:
                if adjusted_epoch >= start_epoch:
                    new_lr = float(lr_policy[start_epoch])
            if new_lr is not None and new_lr != last_lr:
                if config.verbose:
                    logger.info(f"Epoch {adjusted_epoch}: Changed learning rate from {last_lr} to {new_lr}")
                last_lr = new_lr
                config.hyperparameters.optimizer.params.lr = new_lr
            adjust_learning_rate(optimizer, last_lr)

        return lr_lambda
    return None


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, pbar=None):
    model.train()
    total_correct, total_loss, batch_count = 0, 0, 0
    # Determine the number of images to log per epoch
    num_images_to_log =  config.num_images_to_log if hasattr(config, 'num_images_to_log') else 1

    # Randomly select unique batches to log images from
    batches_to_log = random.sample(range(len(train_loader)), min(num_images_to_log, len(train_loader)))

    for batch_idx, (train_x_batch, train_y_batch) in enumerate(train_loader, 1):
        train_x_batch, train_y_batch = train_x_batch.to(device), train_y_batch.to(device)
        optimizer.zero_grad()
        logits = model(train_x_batch)
        loss = loss_fn(logits, train_y_batch)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            #visualize the filters in the first convolutional layer
            layer = conv_layer_to_vis(config.conv_layer_to_vis, model) if hasattr(config, 'conv_layer_to_vis') else None
            if layer is not None:
                #I need to find a way to know what epoc
                draw_conv_filters(epoch, batch_idx, layer, config.conv_layer_to_vis, config.save_dir)

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == train_y_batch).sum().item()
        batch_count += train_x_batch.size(0)

        if batch_idx % config.log_interval == 0 and pbar is not None:
            pbar.set_postfix({'Loss': total_loss / batch_count, 'Accuracy': total_correct / batch_count * 100})
            pbar.update(50)

        # Log to wandb
        if wandb is not None:
            wandb.log({'Train/epoch': epoch, 'Batch': batch_idx, 'Train/Loss': total_loss / batch_count, 'Train/accuracy': total_correct / batch_count * 100})
            # Log images to wandb
            
            if batch_idx in batches_to_log and config.log_images:
                # Log the images
                wandb.log({f'Train/image_epoch_{epoch}_batch_{batch_idx}': wandb.Image(unnormalize(train_x_batch[0], config.mean, config.std), caption=f'Label: {train_y_batch[0].argmax(dim=0)}')})
    pass




##its not a good idea to make the config global becuase it would make it difficult to test and debug the code later in the future. Also, we should only validate if validation data is provided       
def train(model, train_x, train_y, valid_x, valid_y, config, device):
    """
    Train a model using the specified datasets, configuration, and device.

    Args:
    model: The neural network model to train.
    train_x, train_y: Training dataset inputs and labels.
    valid_x, valid_y: Validation dataset inputs and labels.
    config: Configuration object containing training parameters and hyperparameters.
    device: The device (CPU/GPU) to use for training.

    Returns:
    None
    """
    hyperparams = config.hyperparameters
    #use the data loader to load the training and validation data
    train_dataset = CIFARDataLoader(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True)

    #get the hyperparameters from the config file
    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    best_val_accuracy = 0
    best_loss = float('inf')

    #get the scheduler if it exists
    if hyperparams.use_scheduler:
        scheduler = get_lr_adjuster(config, optimizer)
        scheduler(0)
    
    #this code was provided in the lab guide to plot the training progress
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    for epoch in range(1, hyperparams.num_epochs + 1):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{hyperparams.num_epochs}", unit='batches') as pbar:
            
            train_epoch(model, train_loader, optimizer, loss_function, device, epoch, config, pbar=pbar)
            # Evaluate the model on the train and validation set as requested in the lab guide
            train_loss, train_acc, _, _ = evaluate("Train", train_x, train_y, model, config, device, epoch=epoch)
            val_loss, val_acc, _, _ = evaluate("Validation", valid_x, valid_y, model, config, device, epoch=epoch)
            
            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [val_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [val_acc]
            # Update the learning rate
            if hyperparams.use_scheduler:
                plot_data['lr'] += [optimizer.param_groups[0]['lr']]
                # Compute the new learning rate using your function and adjust it
                scheduler(epoch)
            else:
                plot_data['lr'] += [hyperparams.optimizer.params.lr]

            # Checkpointing logic
            #TODO CHECKPOINT AND SAVE THE BEST MODEL
            # if should_save_checkpoint(current_val_accuracy, best_val_accuracy) and config.save_checkpoint:
            #     best_val_accuracy = current_val_accuracy
            #     save_checkpoint(model, optimizer, epoch, current_val_accuracy, config)
                
            # Early Stopping Check
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter >= config.hyperparameters.patience:
                    logger.info(f'No improvement for {config.hyperparameters.patience} epochs. Stopping the training.')
                    break
    plot_training_progress(plot_data, show=True)
    #close the progress bar and finish the wandb run
    pbar.close()
    best_model = DynamicNet(config, verbose=False)
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state)
    return best_model


def should_save_checkpoint(current_metric, best_metric, improvement_threshold=0.1, mode='max'):
    """
    Determines whether the current model should be saved based on the chosen metric and improvement threshold.

    Args:
    current_metric: The metric value of the current model (e.g., accuracy, F1 score).
    best_metric: The best metric value achieved so far.
    config: Configuration object containing model saving parameters.
    improvement_threshold: The minimum relative improvement required to save a new checkpoint.
    mode: 'max' for metrics where higher is better (like accuracy), 
          'min' for metrics where lower is better (like loss).

    Returns:
    bool: True if the current model should be saved, False otherwise.
    """
    try:
        improvement = (current_metric - best_metric) / best_metric
    except ZeroDivisionError:
        improvement = float('inf')
    
    print('improvement', improvement)

    if mode == 'max':
        return improvement > improvement_threshold
    elif mode == 'min':
        return -improvement > improvement_threshold
    else:
        raise ValueError("mode should be either 'max' or 'min'")

def save_checkpoint(model, optimizer, epoch, val_accuracy, config):
    """
    Saves the model checkpoint.

    Args:
    model: The model to save.
    optimizer: The optimizer used during training.
    epoch: The current epoch number.
    val_accuracy: The validation accuracy.
    config: Configuration object containing the save directory.

    Returns:
    None
    """
    model_path = os.path.join(config.save_dir, f'model_epoch_{epoch}_val_acc_{val_accuracy:.4f}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }, model_path)

    if config.use_wandb:
        shutil.copy(model_path, os.path.join(wandb.run.dir, os.path.basename(model_path)))
        wandb.save(os.path.join(wandb.run.dir, os.path.basename(model_path)))

def initialize_dataloader(x, y, config):
    dataset = CIFARDataLoader(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=config.hyperparameters.batch_size, shuffle=False)

def inspect_data(data):
    for i in range(10):
        plt.imshow(data.data[i])
        plt.title(f"Label: {data.targets[i]}")
        plt.show()

def select_top_incorrect_images(detailed_data, top_k=3):
    incorrect_images = [(img, true_label, logits, loss) for img, true_label, logits, loss in detailed_data if true_label != logits.argmax()]
    sorted_incorrect_images = sorted(incorrect_images, key=lambda x: x[3], reverse=True)
    return sorted_incorrect_images[:top_k]

def visualize_incorrect_predictions(incorrect_images, class_names, mean, std, show=False):
    for img, true_label, logits, loss in incorrect_images:
        probabilities = nn.functional.softmax(logits - logits.max(), dim=0) # use stable softmax to get the probabilities
        top3_probs, top3_preds = probabilities.topk(3)
        img = unnormalize(img, mean, std)
        plt.imshow(img)
        top3_labels = [class_names[idx] for idx in top3_preds]
        plt.title(f"True: {class_names[true_label]}, Top 3 preds: {top3_labels}, Loss: {loss:.2f}")
        if show:
            plt.show()
        if wandb is not None:
            #save the image in the buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image(buf)
            wandb.log({'Incorrect predictions': wandb.Image(image)})
            buf.close()
    pass

def unnormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean.view(3, 1, 1), std.view(3, 1, 1)):
        t.mul_(s).add_(m)
    return ((tensor.permute(1, 2, 0).numpy())*255).astype(np.uint8)

def model_evaluation(model, loader, loss_fn, device, detailed=False):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    total_samples = 0
    detailed_data = []
    with torch.no_grad():
        for batch_num, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            losses = loss_fn(logits, y_batch).cpu()
            batch_loss = losses.sum().item()  # Sum the losses for the current batch
            total_loss += batch_loss
            total_samples += x_batch.size(0)  # Count the number of samples in the batch
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            total_samples += x_batch.size(0)
            if detailed:
                # Collect detailed data for each sample to be used for incorrect prediction visualization
                detailed_data.extend(zip(x_batch.cpu(), y_batch.cpu(), logits.cpu(), losses))
    # return total_loss, all_predictions, all_targets, total_samples, detailed_data
    return total_loss, all_predictions, all_targets, batch_num, detailed_data
                


def eval_perf_multi(Y, Y_):
        assert Y.ndim == 1 and Y_.ndim == 1, "Y and Y_ must be 1D arrays"
        assert len(Y) == len(Y_), "Y and Y_ must be of the same length"
        
        n = max(max(Y), max(Y_)) + 1  # Number of classes
        M = np.zeros((n, n), dtype=int)

        for true_label, predicted_label in zip(Y, Y_):
            M[predicted_label, true_label] += 1

        pr = []
        for i in range(n):
            tp_i = M[i,i]
            fn_i = np.sum(M[i,:]) - tp_i
            fp_i = np.sum(M[:,i]) - tp_i
            tn_i = np.sum(M) - (tp_i + fn_i + fp_i)
            
            recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
            precision_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
            pr.append((precision_i, recall_i))

        accuracy = np.trace(M) / np.sum(M)
        avg_f1_score = np.mean([2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in pr]) # f1 score is the average of the f1 scores of each class
        return accuracy, pr, avg_f1_score, M
    
def plot_confusion_matrix(name, epoch, cm, wandb_instance):
    if wandb_instance is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap(cm, annot=True, ax=ax, fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'{name.capitalize()} Confusion Matrix at Epoch {epoch}')
        wandb_instance.log({f'{name}/confusion_matrix_epoch_{epoch}': wandb.Image(fig)})
        plt.close(fig)

def log_metrics(name, epoch, metrics, wandb_instance):
        if wandb_instance is not None:
            wandb_instance.log({f'{name}/{key}_{epoch}': value for key, value in metrics.items()})

def evaluate(name, x, y, model, config, device, wandb=None, epoch=None):
    loader = initialize_dataloader(x, y, config)
    loss_fn = get_loss_function(config)
    total_loss, all_predictions, all_targets, total_samples, viz_data = model_evaluation(model, loader, loss_fn, device, detailed=config.visualize_incorrect)
    accuracy, precision_recall, avg_f1_score, cm = eval_perf_multi(np.array(all_targets), np.array(all_predictions))
    avg_loss = total_loss / total_samples

    if config.verbose:
        table = [
            [f'Epoch', f'{epoch}' if epoch is not None else 'Test'],
            [f'{name} Samples', f'{total_samples}'],
            [f'{name} Loss', f'{avg_loss :.2f}'],
            [f'{name} Accuracy', f'{accuracy * 100:.2f}'],
            ['Classnames', ', '.join(config.classnames)],
            [f'{name} Precision/Class', ', '.join(f'{p[0] * 100:.2f}' for p in precision_recall)],
            [f'{name} Recall/Class', ', '.join(f'{r[1] * 100:.2f}' for r in precision_recall)],
            [f'{name} Avg F1', f'{avg_f1_score * 100:.2f}']
        ]
        print(tabulate(table, headers=['Metric', 'Value %'], tablefmt='grid'), '\n')

    if config.visualize_incorrect and name.lower() == 'validation' and viz_data != []:
        top_incorrect_images = select_top_incorrect_images(viz_data, top_k=config.num_incorrect_to_visualize if hasattr(config, 'num_incorrect_to_visualize') else 1)
        visualize_incorrect_predictions(top_incorrect_images, config.classnames, config.mean, config.std)

    if epoch is not None: #if epoch is not None then we are evaluating the model on the validation set
        log_metrics(name, epoch, {'loss': avg_loss, 'accuracy': accuracy, 'precision_recall': precision_recall, 'f1': avg_f1_score}, wandb)

    plot_confusion_matrix(name, epoch, cm, wandb)
    return avg_loss, accuracy, precision_recall, cm

def setup_paths(args=None):
    # Set up and return data directory and save directory paths
    SAVE_DIR = Path(__file__).parent / args.save_dir
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return SAVE_DIR

def load_cifar_dataset(SAVE_DIR):
    # Load the CIFAR-10 dataset
    if not os.path.exists(os.path.join(SAVE_DIR, 'CIFAR10')):
        trainset = CIFAR10(root=SAVE_DIR, train=True, download=True, transform=None)
        testset = CIFAR10(root=SAVE_DIR, train=False, download=True, transform=None)
    else:
        trainset = CIFAR10(root=SAVE_DIR, train=True, download=False, transform=None)
        testset = CIFAR10(root=SAVE_DIR, train=False, download=False, transform=None)
    return trainset, testset

def split_dataset(trainset, split_ratio=0.8):
    # Split the dataset into training and validation sets
    train_size = int(split_ratio * len(trainset))
    valid_size = len(trainset) - train_size
    indices = np.arange(len(trainset))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    train_dataset = Subset(trainset.data, train_indices)
    valid_dataset = Subset(trainset.data, valid_indices)
    train_labels = np.array(trainset.targets)[train_indices]
    valid_labels = np.array(trainset.targets)[valid_indices]
    return train_dataset, train_labels, valid_dataset, valid_labels

def normalize_data(data, mean, std):
    # Normalize the data
    return (data - mean) / std

def load_config(args):
    # Load configuration from YAML file
    try:
        with open(args.config_filepath, 'r') as file:
            config = DotDict(yaml.safe_load(file))
        return config
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        raise ConfigurationError("Could not find the configuration file.")

def save_model(model, config):
    if config.hyperparameters.save_best_model and wandb is not None:
        run_dir = wandb.run.dir 
        model_path = os.path.join(run_dir, 'best_model.pt')
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path, base_path=run_dir)
        if config.verbose:
            logger.info(f"Saved model to {model_path}")
    pass

def create_model(config, input_shape, device):
    # Create and return the model
    assert input_shape == tuple(config['input_shape']), f"Config input shape {config['input_shape']} does not match the calculated input shape {input_shape}"
    assert config['num_classes'] == len(config.classnames), f"Config number of classes {config['num_classes']} does not match the number of classes in the dataset {len(config.classnames)}"
    model = DynamicNet(config, verbose=True, device=device)
    return model

def calculate_mean_std(data):
    # Calculate mean and standard deviation of the training set
    mean = data.mean(axis=(0, 2, 3))
    std = data.std(axis=(0, 2, 3))
    return mean.reshape((1, data.shape[1], 1, 1)), std.reshape((1, data.shape[1], 1, 1))

def main(args):
    # torch.manual_seed(int(time.time() * 1e6) % 2**31)
    torch.manual_seed(args.seed)
    
    SAVE_DIR = setup_paths(args)
    trainset, testset = load_cifar_dataset(SAVE_DIR)
    train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(trainset)
    
  
    train_data, train_targets = preprocess_dataset(train_dataset, train_labels)
    valid_data, valid_targets = preprocess_dataset(valid_dataset, valid_labels)
    test_data, test_targets = preprocess_dataset(testset.data, testset.targets)
    
    mean, std = calculate_mean_std(train_data)
    train_data = normalize_data(train_data, mean, std)
    valid_data = normalize_data(valid_data, mean, std)
    test_data = normalize_data(test_data, mean, std)
    config = load_config(args=args)
    config.mean, config.std = mean, std #add the mean and std to the config object so that we can use it later
    config.classnames = trainset.classes #add the class names to the config object
    config.custom_loss_functions = args.custom_loss_functions #add the custom loss functions to the config object
    input_shape = train_data.shape[1:]
    
    # setup_wandb(config)
    device = args.device
    model = create_model(config, input_shape, device)
    model.to(device)
    
    train(model, train_data, train_targets, valid_data, valid_targets, config, device)
    #evaluate the model on the test set
    evaluate("Test", test_data, test_targets, model, config, device, wandb=wandb)
    #save the model
    save_model(model, config)

    #finish the wandb run
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    wandb = None
    #add the multiclass hinge loss to the nn module so that we can use it later in case the user passed it
    # nn.MultiClassHingeLoss = MulticlassHingeLoss
    class Args:
        seed = 7052020
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_filepath = './config.yml'
        save_dir = 'CIFAR10' #the directory where the data is saved
        custom_loss_functions = {
                                'MulticlassHingeLoss': MulticlassHingeLoss()
                                # Add other custom loss functions here
                                }

    args = Args()
    main(args)



#TODO: Train the model. It would be nice to have some sort of mode flag to switch between training and evaluation
#Also to be able to resume training from a checkpoint in wandb
#train the model for the specified number of epochs, if resume is set to True then resume training from the last checkpoint. checkpoint_dir is the directory where the checkpoints are saved. 


