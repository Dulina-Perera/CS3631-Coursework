# _1_Back_Propagation/Solution/neuron.py

# %%
import matplotlib.pyplot as plt
import numpy as np
import sys

from tqdm import tqdm

# %%
np.random.seed(42)

# %%
class Neuron:
  def __init__(self, input_size: int, learning_rate: float) -> None:
    self.weights = np.random.rand(input_size) * 2 - 1
    self.bias = np.random.rand()
    self.learning_rate = learning_rate

  def forward(self, inputs: np.ndarray) -> np.ndarray:
    self.inputs = inputs
    self.output = self.relu(np.dot(self.weights, inputs) + self.bias)
    return self.output

  def backward(self, error: float) -> None:
    self.error = error
    relu_derivative = 1.0 if self.output > 0 else 0.0  # Derivative of ReLU
    self.weights -= self.learning_rate * error * relu_derivative * self.inputs
    self.bias -= self.learning_rate * error * relu_derivative

  def relu(self, x: float) -> float:
    return np.maximum(0, x)

  def __str__(self):
    return f'Neuron(weights={self.weights}, bias={self.bias})'


class DenseLayer:
  def __init__(self, input_size: int, output_size: int, learning_rate: float) -> None:
    self.neurons = [Neuron(input_size, learning_rate) for _ in range(output_size)]

  def forward(self, inputs: np.ndarray) -> np.ndarray:
    return np.array([neuron.forward(inputs) for neuron in self.neurons])

  def backward(self, errors: np.ndarray) -> None:
    for neuron, error in zip(self.neurons, errors):
      neuron.backward(error)

  def __str__(self):
    return f'Layer(neurons={self.neurons})'


if __name__ == '__main__':
  # Define the neural network
	layer1 = DenseLayer(2, 3, 0.1)
	layer2 = DenseLayer(3, 1, 0.1)

	# Define the dataset
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	y = np.array([0, 1, 1, 0])

	# Train the neural network
	for epoch in tqdm(range(1000)):
		for i in range(len(X)):
			x = X[i]
			target = y[i]

			# Forward pass
			output1 = layer1.forward(x)
			output2 = layer2.forward(output1)

			# Calculate the loss
			loss = (output2[0] - target) ** 2

			# Backward pass
			layer2.backward(2 * (output2 - target))
			layer1.backward(layer2.neurons[0].error * layer2.neurons[0].weights)

	# Test the neural network
	for i in range(len(X)):
		x = X[i]
		target = y[i]

		# Forward pass
		output1 = layer1.forward(x)
		output2 = layer2.forward(output1)

		print(f'Input: {x}, Target: {target}, Prediction: {output2[0]}')
