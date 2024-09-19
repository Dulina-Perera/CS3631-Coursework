# _1_Back-Propagation/code/loss/loss_fn.py

# %%
import numpy as np

from nptyping import NDArray

# %%
def cat_cross_entropy(
	y_true: NDArray,  # The true labels (ground truth)
	y_pred: NDArray  # The predicted probabilities
) -> NDArray:
	"""
	Calculates the categorical cross-entropy loss between the true labels and predicted probabilities.

	Parameters:
	- y_true: NDArray, shape (batch_size, num_classes)
		The true labels (ground truth) for each sample in the batch.
	- y_pred: NDArray, shape (batch_size, num_classes)
		The predicted probabilities for each class for each sample in the batch.

	Returns:
	- loss: NDArray, shape (batch_size,)
		The calculated loss for each sample in the batch.
	"""
	y_pred_clipped: NDArray = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predicted probabilities to avoid log(0) errors
	return -np.sum(y_true * np.log(y_pred_clipped), axis=1)  # Calculate cross-entropy loss for each sample
