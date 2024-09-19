# _1_Back-Propagation/code/nn.py

# %%
import numpy as np

from layer import ActivationLayer, DenseLayer, relu, relu_prime, softmax
from nptyping import NDArray
from typing import Callable, List, Union

# %%
class NeuralNetwork:
  def __init__(self) -> None:
    self.layers = []
    self.loss_fn = None
    self.loss_prime = None
    self.err_hist = []

  def add(
    self,
    layer: Union[ActivationLayer, DenseLayer]
  ) -> None:
    if isinstance(layer, ActivationLayer) or isinstance(layer, DenseLayer):
      self.layers.append(layer)

  def use(
    self,
    loss_fn: Callable,
    loss_prime: Callable
	) -> None:
    if callable(loss_fn) and callable(loss_prime):
      self.loss_fn = loss_fn
      self.loss_prime = loss_prime

  def train(
		self,
		X_train: NDArray,
		y_train: NDArray,
		epochs: int,
		patience: int
	) -> None:
    patience_counter: int = 0
    least_err: float = float('inf')
    best_weights: Union[List, None] = None
    best_biases: Union[List, None] = None

    for epoch in range(epochs):
      err: float = 0
      for (X, y) in zip(X_train, y_train):
        output: NDArray = X
        for layer in self.layers:
          output = layer.forward(output)

        err += self.loss_fn(y, output)

        output_grad: NDArray = self.loss_prime(y, output)
        for layer in reversed(self.layers):
          output_grad = layer.backward(output_grad)

      err /= len(X_train)
      self.err_hist.append(err)
      print(f'Epoch {epoch + 1}/{epochs} | Error: {err}')

      if err <= least_err:
        patience_counter = 0
        least_err = err
        best_weights = [layer.weights for layer in self.layers if hasattr(layer, 'weights')]
        best_biases = [layer.biases for layer in self.layers if hasattr(layer, 'biases')]
      else:
        patience_counter += 1

      if patience_counter == patience:
        print(f"Early stopping on epoch {epoch + 1} due to no improvement in error for {patience} consecutive epochs.")

        if best_weights and best_biases:
          for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
              layer.weights = best_weights[i]
            if hasattr(layer, 'biases'):
              layer.biases = best_biases[i]

        break
