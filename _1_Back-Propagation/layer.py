# _1_Back-Propagation/dense_layer.py

# %%
import numpy as np

# %%
class DenseLayer:
  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size
    self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
    self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

  def forward(self, input):
    self.input = input
    return np.dot(input, self.weights) + self.bias

  def backward(self, output_gradient, learning_rate):
    if len(self.input.shape) == 1:
      self.input = self.input.reshape(1, -1)

    if len(output_gradient.shape) == 1:
      output_gradient = output_gradient.reshape(1, -1)

    input_error = np.dot(output_gradient, self.weights.T)
    weights_gradient = np.dot(self.input.T, output_gradient)

    self.weights -= learning_rate * weights_gradient
    self.bias -= learning_rate * output_gradient

    return input_error

class ActivationLayer:
  def __init__(self, activation_function, activation_prime):
    self.activation_function = activation_function
    self.activation_prime = activation_prime

  def forward(self, input_data: np.ndarray) -> np.ndarray:
    self.input = input_data
    return self.activation_function(self.input)

  def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    if self.activation_prime is None:
      return output_gradient
    return output_gradient * self.activation_prime(self.input)
