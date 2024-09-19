# _1_Back-Propagation/network,py

# %%
# Import necessary libraries and modules.
import numpy as np

from layer import ActivationLayer, DenseLayer, relu, relu_prime, softmax
from loss import cat_cross_entropy, cat_cross_entropy_prime
from nn import NeuralNetwork
from nptyping import NDArray

# %%
EPOCHS: int = 100
LR: float = 1

# %%
if __name__ == '__main__':
  # Create a neural network model.
  network: NeuralNetwork = NeuralNetwork()

	# Add layers to the neural network model.
  network.add(DenseLayer(14, 100, LR))
  network.add(ActivationLayer(relu, relu_prime))
  network.add(DenseLayer(100, 40, LR))
  network.add(ActivationLayer(relu, relu_prime))
  network.add(DenseLayer(40, 4, LR))
  network.add(ActivationLayer(softmax, None))

  # Define the loss function and its derivative to be used in the neural network model.
  network.use(cat_cross_entropy, cat_cross_entropy_prime)

	# Load the datasets.
  X_train: NDArray = np.genfromtxt(
		fname='./data/x_train.csv',
		dtype=np.int8,
  	delimiter=','
	)
  y_train: NDArray = np.genfromtxt(
		fname='./data/y_train.csv',
		dtype=np.int8,
		delimiter=','
	)
  X_test: NDArray = np.genfromtxt(
		fname='./data/x_test.csv',
		dtype=np.int8,
		delimiter=','
	)
  y_test: NDArray = np.genfromtxt(
		fname='./data/y_test.csv',
		dtype=np.int8,
		delimiter=','
	)

  # One-hot encode the target variable.
  num_classes: int = 4
  y_train_one_hot: NDArray = np.eye(
		N=num_classes,
		dtype=np.int8
	)[y_train - 1]
  y_test_one_hot: NDArray = np.eye(
		N=num_classes,
		dtype=np.int8
	)[y_test - 1]

	# Train the neural network model.
  network.train(X_train, y_train_one_hot, EPOCHS, patience=10)
