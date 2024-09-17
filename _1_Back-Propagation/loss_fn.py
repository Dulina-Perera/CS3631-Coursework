# _1_Back-Propagation/loss_fn.py

# %%
import numpy as np

# %%
class LossFunction:
  @staticmethod
  def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred_clipped), axis=1)

  @staticmethod
  def categorical_cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true
