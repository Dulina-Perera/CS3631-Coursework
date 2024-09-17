# _1_Back-Propagation/network,py

# %%
import numpy as np

from activation_fn import ActivationFunction
from loss_fn import LossFunction
from layer import ActivationLayer, DenseLayer
from nn import NeuralNetwork

# %%
if __name__ == '__main__':
  X_train = np.genfromtxt('Resources/x_train.csv', delimiter=',')
  y_train = np.genfromtxt('Resources/y_train.csv', delimiter=',')
  X_test = np.genfromtxt('Resources/x_test.csv', delimiter=',')
  y_test = np.genfromtxt('Resources/y_test.csv', delimiter=',')

  num_classes = 4
  y_train_one_hot = np.eye(num_classes)[(y_train - 1).astype(int)]

  network = NeuralNetwork()

  network.add(DenseLayer(14, 100))
  network.add(ActivationLayer(ActivationFunction.relu, ActivationFunction.relu_prime))
  network.add(DenseLayer(100, 40))
  network.add(ActivationLayer(ActivationFunction.relu, ActivationFunction.relu_prime))
  network.add(DenseLayer(40, 4))
  network.add(ActivationLayer(ActivationFunction.softmax, None))

  network.use(LossFunction.categorical_cross_entropy, LossFunction.categorical_cross_entropy_prime)

  epochs = 100
  learning_rate = 0.01
  network.train(X_train, y_train_one_hot, epochs, learning_rate)
