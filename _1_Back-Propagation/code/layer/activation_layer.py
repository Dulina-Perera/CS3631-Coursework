# _1_Back-Propagation/code/layer/activation_layer.py

# %%
from nptyping import NDArray
from typing import Callable

# %%
class ActivationLayer:
	"""
	A class representing an activation layer in a neural network

	Attributes:
	-----------
	activation_fn : Callable
		The activation function to apply to the input
	activation_prime : Callable
		The derivative of the activation function
  input : NDArray
		The input data for the layer

	Methods:
	--------
	forward(input):
		Performs the forward pass, computing the output for the given input.
	backward(output_gradient, learning_rate):
		Performs the backward pass, computing the gradient to propagate to the previous layer.
	"""
	def __init__(
		self,
		activation_fn: Callable,
		activation_prime: Callable
	) -> None:
		"""
		Initializes the ActivationLayer with the given activation function and its derivative.

		Parameters:
		-----------
		activation_fn : Callable
			The activation function to apply to the input
		activation_prime : Callable
			The derivative of the activation function
		"""
		# Store the activation function and its derivative.
		self.activation_fn = activation_fn
		self.activation_prime = activation_prime

	def forward(
		self,
		input: NDArray
	) -> NDArray:
		"""
		Performs the forward pass through the layer, applying the activation function to the input.

		Parameters:
		-----------
		input : NDArray
			The input data for the layer

		Returns:
		--------
		NDArray
			The output from the layer
		"""
		# Store the input for use in the backward pass.
		self.input = input

		# Apply the activation function to the input.
		return self.activation_fn(input)

	def backward(
		self,
		output_gradient: NDArray
	) -> NDArray:
		"""
		Performs the backward pass, computing the gradient to propagate to the previous layer.

		Parameters:
		-----------
		output_gradient : NDArray
			The gradient to propagate back through the layer

		Returns:
		--------
		NDArray
			The gradient to propagate to the previous layer
		"""
		# Compute the gradient to propagate back through the layer.
		if self.activation_prime is None:
			return output_gradient
		else:
			return output_gradient * self.activation_prime(self.input)
