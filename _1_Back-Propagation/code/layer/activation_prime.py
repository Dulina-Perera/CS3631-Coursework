# _1_Back-Propagation/code/layer/activation_prime.py

# %%
import numpy as np

from nptyping import NDArray

# %%
def relu_prime(x: NDArray) -> NDArray:
	"""
	Computes the derivative of the Rectified Linear Unit (ReLU) activation function element-wise.

	Parameters:
	-----------
	x (NDArray): Input array

	Returns:
	--------
	NDArray: Output array after applying the derivative of the ReLU activation function
	"""
	return np.where(x > 0, 1, 0)
