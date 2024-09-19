# _1_Back-Propagation/network,py

# %%
# Import necessary libraries and modules.
import numpy as np

from activation_function import ActivationFunction
from layer import ActivationLayer, DenseLayer
from loss_function import LossFunction
from nn import NeuralNetwork
from nptyping import Int8, NDArray

# %%
if __name__ == '__main__':
  # Create a neural network model.
  network: NeuralNetwork = NeuralNetwork()

	# Add layers to the neural network model.
  network.add(DenseLayer(14, 100))
  network.add(ActivationLayer(ActivationFunction.relu, ActivationFunction.relu_prime))
  network.add(DenseLayer(100, 40))
  network.add(ActivationLayer(ActivationFunction.relu, ActivationFunction.relu_prime))
  network.add(DenseLayer(40, 4))
  network.add(ActivationLayer(ActivationFunction.softmax, None))

	# -----------------------------------------------------------------------------------------------
  # # Load the datasets.
  # X_train: NDArray[NDArray[Int8]] = np.genfromtxt(
	# 	fname='./Resources/x_train.csv',
	# 	dtype=np.int8,
  # 	delimiter=','
	# )
  # y_train: NDArray[Int8] = np.genfromtxt(
	# 	fname='./Resources/y_train.csv',
	# 	dtype=np.int8,
	# 	delimiter=','
	# )
  # X_test: NDArray[NDArray[Int8]] = np.genfromtxt(
	# 	fname='./Resources/x_test.csv',
	# 	dtype=np.int8,
	# 	delimiter=','
	# )
  # y_test: NDArray[Int8] = np.genfromtxt(
	# 	fname='./Resources/y_test.csv',
	# 	dtype=np.int8,
	# 	delimiter=','
	# )

  # # One-hot encode the target variable.
  # num_classes: int = 4
  # y_train_one_hot: NDArray[NDArray[Int8]] = np.eye(
	# 	N=num_classes,
	# 	dtype=np.int8
	# )[y_train - 1]
  # y_test_one_hot: NDArray[NDArray[Int8]] = np.eye(
	# 	N=num_classes,
	# 	dtype=np.int8
	# )[y_test - 1]

	# # Define the loss function and its derivative to be used in the neural network model.
  # network.use(LossFunction.categorical_cross_entropy, LossFunction.categorical_cross_entropy_prime)

	# # Train the neural network model.
  # epochs: int = 100
  # learning_rate: float = 1
  # network.train(X_train, y_train_one_hot, epochs, learning_rate, 100)

  # # Plot the error history.
  # network.plot_error(f'plots/{epochs}_{learning_rate}.png')

  # # Load the weights and biases from the trained neural network model.
  # # network.load_wandb('wandb')

  # network.save_wandb('wandb')

  # -----------------------------------------------------------------------------------------------
  network.load_wandb('a')
  network.train(
		np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1],]).reshape(1, -1),
		np.array([[0, 0, 0, 1]]),
		epochs=1,
		learning_rate=0.1
	)

