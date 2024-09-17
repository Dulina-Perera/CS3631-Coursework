# _1_Back-Propagation/nn.py

# %%
import numpy as np

# %%
class NeuralNetwork:
  def __init__(self):
    self.layers = []
    self.loss = None
    self.loss_prime = None

  def add(self, layer):
    self.layers.append(layer)

  def use(self, loss, loss_prime):
    self.loss = loss
    self.loss_prime = loss_prime

  def predict(self, input_data: np.ndarray) -> np.ndarray:
    samples = len(input_data)
    result = []

    for i in range(samples):
      output = input_data[i]
      for layer in self.layers:
        output = layer.forward(output)
      result.append(output)

      return np.array(result)

  def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float):
    for epoch in range(epochs):
      error = 0
      for x, y in zip(x_train, y_train):
        output = x
        for layer in self.layers:
          output = layer.forward(output)

        error += np.mean(self.loss(y, output))

        output_gradient = self.loss_prime(y, output)
        for layer in reversed(self.layers):
          output_gradient = layer.backward(output_gradient, learning_rate)

      error /= len(x_train)
      print(f'Epoch {epoch + 1}/{epochs}, Error: {error}')
