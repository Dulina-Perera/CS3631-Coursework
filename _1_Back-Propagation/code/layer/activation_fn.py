# _1_Back-Propagation/code/layer/activation_fn.py

# %%
import numpy as np

from nptyping import NDArray

# %%
def relu(x: NDArray) -> NDArray:
	"""
	Applies the Rectified Linear Unit (ReLU) activation function element-wise to the input array.

	Parameters:
	x (NDArray): Input array

	Returns:
	NDArray: Output array after applying the ReLU activation function
	"""
	return np.maximum(0, x)

def softmax(x: NDArray) -> NDArray:
	"""
	Applies the Softmax activation function to the input array.

	Parameters:
	x (NDArray): Input array

	Returns:
	NDArray: Output array after applying the Softmax activation function
	"""
	if len(x.shape) == 1: # For 1D input
		exp_values = np.exp(x)
		return exp_values / np.sum(exp_values)
	else: # For 2D input
		exp_values = np.exp(x)
		return exp_values / np.sum(exp_values, axis=1, keepdims=True)
