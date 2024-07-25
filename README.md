# Convolutional Neural Networks for Image Classification

## Project Overview

This project explores the implementation and training of Convolutional Neural Networks (CNNs) for image classification tasks using the MNIST and CIFAR-10 datasets. It involves a series of tasks to progressively build a deeper understanding of CNNs and their applications.

---

## Project Objectives

1. Implement Convolutional Layers: Forward and backward passes for fully connected layers, ReLU activation, and softmax cross-entropy loss.
2. Incorporate Regularization Techniques: Add L2 regularization and evaluate its effects.
3. Use PyTorch for Model Implementation: Recreate and train the model using PyTorch.
4. Extend Model Training to CIFAR-10: Adapt the model architecture for CIFAR-10.
5. Visualize and Interpret Model Training: Log training progress using Weights & Biases (W&B).
6. Explore Alternative Loss Functions: Implement and experiment with a multiclass hinge loss.

---

## Project Structure

- **setup_cython.py**: Compiles Cython extensions.
- **task2.py**: Trains the neural network on MNIST with L2 regularization.
- **task3.py**: Logs training results to W&B.
- **task4.py**: Trains the model on CIFAR-10.
- **task4_bonus.py**: Implements multiclass hinge loss.
- **train.py**: Trains the basic CNN model on MNIST.
- **layers.py**: Layer definitions and implementations.
- **nn.py**: Neural network training and evaluation functions.
- **check_grads.py**: Checks gradients for debugging.
- **lab2.pdf**: Detailed instructions and explanations.

---

## Setup Instructions

1. **Clone the Repository:**
   ```sh
   git clone <repository_url>
   cd cnn-image-classification
   ```

2. **Set Up the Python Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Compile the Cython Extensions:**
   ```sh
   python setup_cython.py build_ext --inplace
   ```

---

## Usage Instructions

### Task 1: Basic Training on MNIST

- **Script:** `train.py`
- **Run:** 
  ```sh
  python train.py
  ```

### Task 2: Training with L2 Regularization

- **Script:** `task2.py`
- **Run:** 
  ```sh
  python task2.py
  ```

### Task 3: PyTorch Implementation

- **Script:** `task3.py`
- **Run:** 
  ```sh
  python task3.py
  ```

### Task 4: Training on CIFAR-10

- **Script:** `task4.py`
- **Run:** 
  ```sh
  python task4.py
  ```

### Bonus Task: Multiclass Hinge Loss

- **Script:** `task4_bonus.py`
- **Run:** 
  ```sh
  python task4_bonus.py
  ```

---

## Notes

- Ensure that the MNIST and CIFAR-10 datasets are available in the specified directories.
- Modify configuration files as needed to suit your experimental setup.
- Detailed instructions and explanations are provided in `lab2.pdf`.

---

## Troubleshooting

- **Cython Compilation Issues:** Ensure you have Cython installed and properly configured.
- **Dataset Paths:** Verify dataset paths in the scripts to match your local setup.
- **W&B Issues:** Refer to the [W&B documentation](https://docs.wandb.ai/) for setup and usage instructions.

---

## Example `requirements.txt`

```txt
numpy
torch
torchvision
matplotlib
tqdm
wandb
skimage
scipy
cython
tabulate
```

---

By following these instructions, you can set up the environment, understand the project structure, and run the code efficiently. This concise documentation ensures clarity and ease of use for all users.

---