import numpy as np
import time
from collections import defaultdict


class RandomNumberGenerator(np.random.RandomState):
    """
    A class for creating the same random samples for a given seed value
    """

    def __init__(self, seed=None):
        self._seed = seed
        super(RandomNumberGenerator, self).__init__(self._seed)

    def reseed(self):
        if self._seed is not None:
            self.seed(self._seed)


class CustomKFold(object):
    """
    A class for splitting into k-fold indices, used only for batch iterations
    Note that this is NOT the same as StratifiedKFold by sklearn.
    It returns only one random permutation of indices.
    """

    def __init__(self, shuffle=False, random_seed=None):
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.rng = RandomNumberGenerator(self.random_seed)

    def make_k_folds(self, y, n_folds=3):
        self.rng.reseed()

        # group indices
        labels_indices = dict()
        for index, label in enumerate(y):
            if isinstance(label, np.ndarray):
                label = tuple(label.tolist())
            if label not in labels_indices:
                labels_indices[label] = list()
            labels_indices[label].append(index)

        # split all indices label-wisely
        for label, indices in sorted(labels_indices.items()):
            labels_indices[label] = np.array_split(indices, n_folds)

        # collect respective splits into folds and shuffle if needed
        for k in range(n_folds):
            fold = np.concatenate(
                [indices[k] for _, indices in sorted(labels_indices.items())]
            )
            if self.shuffle:
                self.rng.shuffle(fold)
            yield fold


class Stopwatch(object):
    """
    A little helper class I referenced from https://bit.ly/2Kr7NVM
    Keeps track of time elapsed while training process.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timerfunc = time.time
        self.start_ = None
        self.elapsed_ = None

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop().elapsed()
        if self.verbose:
            print("Elapsed time: {0:.3f} sec".format(elapsed))

    def start(self):
        self.start_ = self.timerfunc()
        self.elapsed_ = None
        return self

    def stop(self):
        self.elapsed_ = self.timerfunc() - self.start_
        self.start_ = None
        return self

    def elapsed(self):
        if self.start_ is None:
            return self.elapsed_
        return self.timerfunc() - self.start_


class RMSProp(object):
    """
    Logic referenced from https://bit.ly/33N38oA and https://bit.ly/2qTluWr
    Code inspired from Keras's optimizers: https://bit.ly/33L4kZE
    """

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        beta=0.999,
        epsilon=1e-8,
        max_epochs=100,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.rho = rho
        self.beta = beta
        self.epsilon = epsilon
        self.t = 1
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.loss_history = []
        self.score_history = []
        self.val_loss_history = []
        self.val_score_history = []
        self.epoch = 0
        self.total_epochs = 0
        self.weights = defaultdict(dict)
        self.decay_weights = defaultdict(dict)

    def setup(self, neural_network):
        for i, layer in enumerate(neural_network.parametric_layers()):
            self.weights[i]["W"] = np.zeros_like(layer.W)
            self.weights[i]["b"] = np.zeros_like(layer.b)
            self.decay_weights[i]["W"] = np.zeros_like(layer.W)
            self.decay_weights[i]["b"] = np.zeros_like(layer.b)

    def update(self, neural_network):
        for i, layer in enumerate(neural_network.parametric_layers()):
            self.weights[i]["W"] = (
                self.rho * self.weights[i]["W"] + (1.0 - self.rho) * layer.dW
            )
            self.weights[i]["b"] = (
                self.rho * self.weights[i]["b"] + (1.0 - self.rho) * layer.db
            )
            self.decay_weights[i]["W"] = (
                self.beta * self.decay_weights[i]["W"]
                + (1.0 - self.beta) * layer.dW ** 2
            )
            self.decay_weights[i]["b"] = (
                self.beta * self.decay_weights[i]["b"]
                + (1.0 - self.beta) * layer.db ** 2
            )
            lr = (
                self.learning_rate
                * np.sqrt(1.0 - self.beta ** self.t)
                / (1.0 - self.rho ** self.t)
            )
            W_step = (
                lr
                * self.weights[i]["W"]
                / (np.sqrt(self.decay_weights[i]["W"]) + self.epsilon)
            )
            b_step = (
                lr
                * self.weights[i]["b"]
                / (np.sqrt(self.decay_weights[i]["b"]) + self.epsilon)
            )
            layer.W -= W_step
            layer.b -= b_step
        self.t += 1

    def train_epoch(self, neural_network):
        self.setup(neural_network)
        losses = []
        for X_batch, y_batch in neural_network.batch_iteration():
            loss = np.mean(neural_network.update(X_batch, y_batch))
            self.update(neural_network)
            neural_network.max_norm_update()
            losses.append(loss)
        return losses  # epoch losses

    def optimize(self, neural_network):
        timer = Stopwatch(verbose=False).start()
        self.total_epochs += self.max_epochs
        for i in range(self.max_epochs):
            self.epoch += 1
            if self.verbose:
                print("Epoch {0}/{1}".format(self.epoch, self.total_epochs))
            losses = self.train_epoch(neural_network)
            self.loss_history.append(losses)
            summary = "–– Time elapsed: {0:.2f} sec\n".format(timer.elapsed())
            summary += "–– Training loss: {0:.3f}\n".format(np.mean(losses))

            score = neural_network.metric(neural_network.y, neural_network.validate())
            self.score_history.append(score)

            summary += "–– Training accuracy: {0:.4f}\n".format(score)
            if neural_network.X_val is not None:
                val_loss = neural_network.loss_function(
                    neural_network.y_val,
                    neural_network.validate_proba(neural_network.X_val),
                )
                self.val_loss_history.append(val_loss)
                val_score = neural_network.metric(
                    neural_network.y_val, neural_network.validate(neural_network.X_val)
                )
                if self.epoch > 1 and val_score < 0.2 * self.val_score_history[-1]:
                    return
                self.val_score_history.append(val_score)
                summary += "–– Validation loss: {0:.3f}\n".format(val_loss)
                summary += "–– Validation accuracy: {0:.4f}\n".format(val_score)
            if self.verbose:
                print(summary)


def initialize_weights(shape, random_seed=None):
    """
    Initialize weights for layers
    :param shape: dimensions
    :param random_seed: random state of seed
    :return: a numpy array of weights
    """
    receptive_field_size = np.prod(shape[2:])
    input_field_size = receptive_field_size * shape[1]
    output_field_size = receptive_field_size * shape[0]
    s = np.sqrt(6.0 / (input_field_size + output_field_size))
    return RandomNumberGenerator(random_seed).uniform(low=-s, high=s, size=shape)


def softmax(z, derivative=False):
    """
    A simple activation function that normalizes input into a probability distribution
    consisting of K probabilities proportional to the exponents of the input
    :param z: input
    :param derivative: flag whether to apply derivative
    :return: probability distribution np array
    """
    z = np.atleast_2d(z)
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    y = e / np.sum(e, axis=1, keepdims=True)
    if derivative:
        return y * (1.0 - y)  # element-wise
    return y


def categorical_crossentropy(y_actual, y_predicted, eps=1e-15, normalize=True):
    """
    :param y_actual: the actual values of y
    :param y_predicted: the predicted values of y
    :param eps: fixed epsilon value
    :param normalize: flag to check whether to normalize
    :return: loss value
    """
    if not isinstance(y_actual, np.ndarray):
        y_actual = np.asarray(y_actual)
    if not isinstance(y_predicted, np.ndarray):
        y_predicted = np.asarray(y_predicted)
    y_predicted = np.clip(y_predicted, eps, 1.0 - eps)
    loss = -np.sum(y_actual * np.log(y_predicted))
    if normalize:
        loss /= float(len(y_actual))
    return loss


def relu(z, derivative=False):
    """
    An activation function that returns values between 0-1
    :param z: input
    :param derivative: flag
    :return: probability values
    """
    if derivative:
        z = np.asarray(z)
        d = np.zeros_like(z)
        d[z > 0] = 1.0
        return d
    return np.maximum(0.0, z)


def accuracy_score(y_actual, y_predicted, normalize=True):
    """
    :param y_actual: actual values of y
    :param y_predicted: predicted values of y
    :param normalize: flag to normalize output or not
    :return: score for model
    """
    if not isinstance(y_actual, np.ndarray):
        y_actual = np.asarray(y_actual)
    if not isinstance(y_predicted, np.ndarray):
        y_predicted = np.asarray(y_predicted)
    score = sum(np.all(a == p) for a, p in zip(y_actual, y_predicted))
    if normalize:
        score /= float(len(y_actual))
    return score


def one_hot_decision_function(y):
    """
    A little helper function that returns predicted values as one hot encodes
    :param y: predicted values of y
    :return: one hot encoded y as per maximum likelihood of a class
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z


def to_categorical(y, num_classes=None, dtype="float32"):
    """
    Converts a class vector (integers) to binary class matrix.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def calculate_loss_gradient(actual, predicted):
    return -(actual - predicted)
