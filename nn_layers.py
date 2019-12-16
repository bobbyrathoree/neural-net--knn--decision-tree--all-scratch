from nn_utils import initialize_weights, relu, RandomNumberGenerator
import numpy as np


class Dense(object):
    """
    Basic dense/fully_connected layer implementation
    """

    def __init__(self, output_dim, bias=1.0, max_norm=-1, random_seed=None):
        self.output_dim = output_dim
        self.bias = bias
        self.init = initialize_weights
        self.max_norm = max_norm
        self.W = np.array([])  # weights will be updated by RMSprop optimizer
        self.b = np.array([])
        self.dW = np.array([])  # dW, db will be used by RMSprop optimizer
        self.db = np.array([])
        self.last_input = None
        self.random_seed = random_seed

    def max_norm_update(self):
        L = np.linalg.norm(self.W, np.inf)
        if L > self.max_norm > 0:
            self.W *= self.max_norm / L

    def setup_weights(self, x_shape):
        self.W = self.init(
            shape=(x_shape[1], self.output_dim), random_seed=self.random_seed
        )
        self.b = np.full(self.W.shape[1], self.bias)

    def forward_pass(self, x):
        self.last_input = x
        return np.dot(x, self.W) + self.b

    def backward_pass(self, residual):
        self.dW = np.dot(self.last_input.T, residual)
        self.db = np.sum(residual, axis=0)
        return np.dot(residual, self.W.T)

    def shape(self, prev_shape):
        return prev_shape[1], self.output_dim


class Activation(object):
    """
    Activation layer intuition with forward and backward pass implementation
    """

    def __init__(self, activation, random_seed=None):
        self.activation_name = activation
        self.activation = relu
        self._last_input = None
        self.random_seed = random_seed

    def forward_pass(self, x):
        self._last_input = x
        return self.activation(x)

    def backward_pass(self, residual):
        return self.activation(self._last_input, derivative=True) * residual

    def setup_weights(self, x_shape):
        pass

    @staticmethod
    def shape(prev_shape):
        return prev_shape


class Dropout(object):
    """
    A dropout layer implementation that randomly sets
    a fraction of `p` inputs to 0 at each training update.
    """

    def __init__(self, p=0.2, random_seed=None, **params):
        self.p = p
        self.is_training = False
        self._mask = None
        self.random_seed = random_seed

    def forward_pass(self, X):
        if self.is_training:
            self._mask = (
                RandomNumberGenerator(self.random_seed).uniform(size=X.shape) > self.p
            )
            Z = self._mask * X
        else:
            Z = (1.0 - self.p) * X  # to keep output of the same scale (on average)
        return Z

    def backward_pass(self, residual):
        return self._mask * residual

    def setup_weights(self, x_shape):
        pass

    @staticmethod
    def shape(prev_shape):
        return prev_shape
