# _1_Back-Propagation/nn.py

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from layer import ActivationLayer, DenseLayer
from nptyping import Int8, Float64, NDArray
from typing import Callable, List, Union

# %%
class NeuralNetwork:
  def __init__(self) -> None:
    self.layers: List[ActivationLayer | DenseLayer] = []
    self.loss: Union[Callable[[NDArray[Int8], NDArray[NDArray[Float64]]], NDArray[Float64]], None] = None
    self.loss_prime: Union[Callable[[NDArray[Int8], NDArray[NDArray[Float64]]], NDArray[NDArray[Float64]]], None] = None
    self.error_history: List[float] = []

  def add(self, layer: Union[ActivationLayer, DenseLayer]) -> None:
    self.layers.append(layer)

  def use(
    self,
    loss: Callable[[NDArray[Int8], NDArray[NDArray[Float64]]], NDArray[Float64]],
    loss_prime: Callable[[NDArray[Int8], NDArray[NDArray[Float64]]], NDArray[NDArray[Float64]]]
  ) -> None:
    self.loss = loss
    self.loss_prime = loss_prime

  def train(
    self,
    X_train: NDArray[NDArray[Int8]],
    y_train: NDArray[Int8],
    epochs: int,
    learning_rate: float,
    patience: int = 10
  ) -> None:
    best_error: float = float('inf')
    patience_counter: int = 0
    best_weights: Union[List[NDArray[NDArray[Float64]]], None] = None
    best_biases: Union[List[NDArray[NDArray[Float64]]], None] = None

    for epoch in range(epochs):
      error: float = 0
      for (x, y) in zip(X_train, y_train):
        output: NDArray[Int8 | Float64] = x
        for layer in self.layers:
          output = layer.forward(output)

        # error += self.loss(y, output)

        output_gradient: NDArray[NDArray[Float64]] = self.loss_prime(y, output)
        for layer in reversed(self.layers):
          output_gradient = layer.backward(output_gradient, learning_rate)

      error /= len(X_train)
      self.error_history.append(error)
      print(f'Epoch {epoch + 1}/{epochs}, Error: {error}')

      if error < best_error:
        best_error = error
        patience_counter = 0

        best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
        best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
      else:
        patience_counter += 1

      if patience_counter >= patience:
        print(f"Early stopping on epoch {epoch + 1} due to no improvement in error for {patience} consecutive epochs.")

        if best_weights is not None:
          for i, layer in enumerate(reversed(self.layers)):
            if hasattr(layer, 'weights'):
              layer.weights = best_weights.pop()
              layer.biases = best_biases.pop()

        print(f'Restored best model with error: {best_error}')
        break

  def save_wandb(self, directory_path: str) -> None:
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)

    for i, layer in enumerate(self.layers):
      if hasattr(layer, 'weights'):
        np.savetxt(f'{directory_path}/weights_layer_{i+1}.txt', layer.weights, delimiter=',')
        np.savetxt(f'{directory_path}/biases_layer_{i+1}.txt', layer.biases, delimiter=',')

  def load_wandb(self, directory_path: str) -> None:
    for i, layer in enumerate(self.layers):
      if hasattr(layer, 'weights'):
        layer.weights = np.loadtxt(f'{directory_path}/weights_layer_{i+1}.txt', delimiter=',')
        layer.biases = np.loadtxt(f'{directory_path}/biases_layer_{i+1}.txt', delimiter=',')

  def predict(self, X_test: NDArray[NDArray[Int8]], y_test: NDArray[Int8] = None) -> Union[float, NDArray]:
    samples = len(X_test)
    correct_predictions = 0
    predictions = []

    for i in range(samples):
      output = X_test[i]
      for layer in self.layers:
        output = layer.forward(output)
      predicted_label = np.argmax(output)
      predictions.append(predicted_label)

      if y_test is not None and predicted_label == np.argmax(y_test[i]):
        correct_predictions += 1

    accuracy = correct_predictions / samples if y_test is not None else None

    if y_test is not None:
      print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy if y_test is not None else np.array(predictions)

  def plot_error(self, file_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(self.error_history) + 1), self.error_history, marker='o')
    plt.title("Training Error vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig(file_path)
    print(f'Error plot saved at {file_path}')
