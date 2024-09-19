# _1_Back-Propagation/code/loss/loss_prime.py

# %%
import numpy as np

from nptyping import NDArray

# %%
def cat_cross_entropy_prime(
  y_true: NDArray,
  y_pred: NDArray
) -> NDArray:
  """
	Calculates the derivative of the categorical cross-entropy loss function.

	Parameters:
		y_true (NDArray): The true labels.
		y_pred (NDArray): The predicted labels.

	Returns:
		NDArray: The derivative of the categorical cross-entropy loss function.
	"""
  return y_pred - y_true
