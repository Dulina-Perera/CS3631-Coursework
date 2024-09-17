# _1_Back-Propagation/Solution/neuron.py

# %%
import warnings; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

from typing import Any

from utils import create_data, plot_data

# %%
np.random.seed(42)

# %%
class Layer_Dense:
  def __init__(self, n_inputs: int, n_neurons: int) -> None:
    self.weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
    self.biases = np.random.rand(n_neurons)

  def forward(self, inputs: np.ndarray) -> None:
    self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs: np.ndarray) -> None:
		self.output = np.maximum(0, inputs)

class Activation_Softmax:
	def forward(self, inputs: np.ndarray) -> None:
		exp_values: Any = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

class Loss:
  def calculate(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
    sample_losses = self.forward(output, y)
    data_loss = np.mean(sample_losses)
    return data_loss

class Loss_CategoricalCrossEntropy(Loss):
	def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
		samples: int = len(y_pred)
		y_pred_clipped: Any = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:
			correct_confidences: Any = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_confidences: Any = np.sum(y_pred_clipped * y_true, axis=1)

		negative_log_likelihoods: Any = -np.log(correct_confidences)
		return negative_log_likelihoods

# %%
if __name__ == '__main__':
  # Define the dataset
	X, y = create_data(100, 3)

	# Define the layers.
	dense1: Layer_Dense = Layer_Dense(2, 5)
	activation1: Activation_ReLU = Activation_ReLU()
	dense2: Layer_Dense = Layer_Dense(5, 3)
	activation2: Activation_Softmax = Activation_Softmax()

	# Make a forward pass of the training data through thes layers.
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	activation2.forward(dense2.output)

	# Calculate the loss from the output of the activation2 layer.
	loss_function: Loss = Loss_CategoricalCrossEntropy()
	loss: np.ndarray = loss_function.calculate(activation2.output, y)
	print('Loss:', loss)
