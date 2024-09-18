# _1_Back-Progation/acitvation_fn.py

# %%
import numpy as np

# %%
class ActivationFunction:
  @staticmethod
  def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

  @staticmethod
  def relu_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

  @staticmethod
  def softmax(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:  # For 1D input
      exp_values = np.exp(x - np.max(x))
      return exp_values / np.sum(exp_values)
    else:  # For 2D input
      exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
      return exp_values / np.sum(exp_values, axis=1, keepdims=True)
