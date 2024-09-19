# _1_Back-Propagation/code/layer/dense_layer.py

# %%
import numpy as np

from nptyping import NDArray

# %%
class DenseLayer:
  """
  A class representing a fully connected (dense) layer in a neural network

  Attributes:
  -----------
  input_size : int
    The size of the input to the layer
  output_size : int
    The size of the output from the layer
  lr : float
		The learning rate used to update the weights and biases
  weights : NDArray
    The weight matrix connecting the inputs to the outputs
  biases : NDArray
    The bias vector added to the output
  input : NDArray
		The input data for the layer (can be a single sample or a batch of samples)

  Methods:
  --------
  forward(input):
    Performs the forward pass, computing the output for the given input.
  backward(output_gradient, lr):
    Performs the backward pass, updating the weights and biases, and computing
    the gradient to propagate to the previous layer.
  """
  def __init__(
		self,
		input_size: int,
		output_size: int,
		lr: float
	) -> None:
    """
    Initializes the DenseLayer with random weights and biases.

    Parameters:
    -----------
    input_size : int
      The size of the input to the layer
    output_size : int
      The size of the output from the layer
    lr : float
			The learning rate used to update the weights and biases
    """
    # Store the input and output sizes.
    self.input_size = input_size
    self.output_size = output_size

    # Initialize the weight matrix with random values, scaled to prevent exploding gradients.
    self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)

    # Initialize the bias vector with random values, similarly scaled.
    self.biases = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    # Store the learning rate for use in the backward pass.
    self.lr = lr

  def forward(
    self,
    input: NDArray
  ) -> NDArray:
    """
    Performs the forward pass for this layer.

    Parameters:
    -----------
    input : NDArray
      The input data for the layer (can be a single sample or a batch of samples)

    Returns:
    --------
    NDArray
      The result of the forward pass: input * weights + biases
    """
    # Store the input to use in the backward pass.
    self.input = input

    # Perform the forward pass (output = input * weights + biases).
    return np.dot(input, self.weights) + self.biases

  def backward(
    self,
    output_gradient: NDArray
	) -> NDArray:
    """
    Performs the backward pass for this layer, updating weights and biases
    based on the gradient of the loss with respect to the output (∂L/∂y).

    Parameters:
    -----------
    output_gradient : NDArray
      The gradient of the loss with respect to the output of this layer (∂L/∂y)

    Returns:
    --------
    NDArray
      The gradient of the loss with respect to the input of this layer (∂L/∂x)
    """
  	# If input is a 1D array, reshape it to a 2D array for matrix operations.
    if len(self.input.shape) == 1:
      self.input = self.input.reshape(1, -1)

    # If output_gradient is a 1D array, reshape it to a 2D array for matrix operations.
    if len(output_gradient.shape) == 1:
      output_gradient = output_gradient.reshape(1, -1)

    # Calculate the gradient of the loss with respect to the input of this layer (for backpropagation).
    input_error: NDArray = np.dot(output_gradient, self.weights.T)

    # Calculate the gradient of the loss with respect to the weights.
    weights_gradient = np.dot(self.input.T, output_gradient)

    # Update the weights by subtracting the gradient multiplied by the learning rate.
    self.weights -= self.lr * weights_gradient

    # Update the biases by subtracting the gradient of the output.
    self.biases -= self.lr * output_gradient

    # Return the gradient of the input, which is propagated backward to the previous layer.
    return input_error
