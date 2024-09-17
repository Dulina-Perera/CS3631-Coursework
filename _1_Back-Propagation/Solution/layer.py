import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input if input.ndim == 2 else input.reshape(1, -1)
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return np.dot(output_error, self.weights.T)

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)

class SoftmaxLayer:
    def forward(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.zeros_like(output_error)
        for i, (output, dvalue) in enumerate(zip(self.output, output_error)):
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            input_error[i] = np.dot(jacobian_matrix, dvalue)
        return input_error

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred

def categorical_crossentropy(y_true, y_pred):
    # Clip predictions to avoid log(0) issues
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

def categorical_crossentropy_prime(y_true, y_pred):
    return y_pred - y_true  # Gradient of cross-entropy w.r.t. softmax output


if __name__ == '__main__':
    network = [
        FCLayer(14, 100),
        ActivationLayer(relu, relu_prime),
        FCLayer(100, 40),
        ActivationLayer(relu, relu_prime),
        FCLayer(40, 4),
        SoftmaxLayer()
    ]

    X_train = np.genfromtxt('../Resources/x_train.csv', delimiter=',')
    y_train = np.genfromtxt('../Resources/y_train.csv', delimiter=',')
    X_test = np.genfromtxt('../Resources/x_test.csv', delimiter=',')
    y_test = np.genfromtxt('../Resources/y_test.csv', delimiter=',')

    epochs = 10000
    for epoch in range(epochs):
        error = 0
        for x, y in zip(X_train, y_train):
            x = x.reshape(1, -1)  # Ensure input is 2D
            output = x
            for layer in network:
                output = layer.forward(output)
            error += categorical_crossentropy(y, output)

            output_error = categorical_crossentropy_prime(y, output)

            for layer in reversed(network):
                output_error = layer.backward(output_error, 0.1)

        error /= len(X_train)
        print(f'Epoch {epoch + 1}/{epochs} - Error: {error}')
