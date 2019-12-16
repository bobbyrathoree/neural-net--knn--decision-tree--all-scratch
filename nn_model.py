from sklearn.model_selection import StratifiedKFold

from nn_process_images import load_data_from_scratch, load_data
from nn_utils import (
    one_hot_decision_function,
    categorical_crossentropy,
    accuracy_score,
    RMSProp,
    CustomKFold,
    to_categorical,
    Stopwatch,
    calculate_loss_gradient,
)
import numpy as np


class NeuralNetwork(object):
    """
    A generic implementation of a neural network where you can
    add any number of layers, given they are either dense, activation,
    followed by a dropout layer.
    """

    def __init__(
        self,
        layers=None,
        n_batches=10,
        optimizer_params=None,
        shuffle=True,
        random_seed=None,
    ):
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.layers = layers
        self.n_batches = n_batches
        self.optimizer_params = optimizer_params
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.loss_function = categorical_crossentropy
        self.loss_grad = calculate_loss_gradient
        self.metric = accuracy_score
        self.optimizer = RMSProp(**self.optimizer_params)
        self.custom_split_function = CustomKFold(
            shuffle=self.shuffle, random_seed=self.random_seed
        )
        self.initialized = False
        self.training = False

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return one_hot_decision_function(y_pred)

    def setup_layers(self, X_shape):
        for layer in self.layers:
            layer.setup_weights(X_shape)  # allocate and initialize
            X_shape = layer.shape(prev_shape=X_shape)  # forward propagate shape
        self.initialized = True

    def forward_pass(self, X_batch):
        Z = X_batch
        for layer in self.layers:
            Z = layer.forward_pass(Z)
        return Z

    def update(self, X_batch, y_batch):
        """
        :param X_batch: X values per batch
        :param y_batch: y values for a batch
        :return: categorical cross-entropy value
        """
        # forward pass
        y_pred = self.forward_pass(X_batch)

        # backward pass
        grad = self.loss_grad(y_batch, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return self.loss_function(y_batch, y_pred)

    def parametric_layers(self):
        """
        Return all the Dense layers of NeuralNetwork
        """
        for layer in self.layers:
            if hasattr(layer, "W"):
                yield layer

    def batch_iteration(self):
        for indices in self.custom_split_function.make_k_folds(
            self.y, n_folds=self.n_batches
        ):
            yield self.X[indices], self.y[indices]

    def fit(self, X, y, X_val=None, y_val=None):
        self.X = X
        self.y = y

        for (k, v) in self.optimizer_params.items():
            setattr(self.optimizer, k, v)

        for layer in self.layers:
            layer.random_seed = self.random_seed

        if not self.initialized:
            self.setup_layers(X.shape)

        self.X_val = X_val
        self.y_val = y_val
        setattr(self.optimizer, "learning_rate", 5e-5)

        if (
            "verbose" in self.optimizer_params
            and self.optimizer_params["verbose"]
            and X_val is not None
        ):
            print(
                "Total samples: {0}\nTraining samples: {1}\nValidating samples: {2}\n".format(
                    len(X) + len(X_val), len(X), len(X_val)
                )
            )
        self.is_training = True
        self.optimizer.optimize(self)
        self.is_training = False

    def minibatch_forward_pass(self, X):
        y_pred = list()
        batch_size = int(len(X) / self.n_batches)
        for i in range(self.n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_pred.append(self.forward_pass(X_batch))
        if self.n_batches * batch_size < len(X):
            y_pred.append(self.forward_pass(X[end:]))
        return np.concatenate(y_pred)

    def validate_proba(self, X=None):
        training_phase = self.is_training
        if training_phase:
            self.is_training = False
        if X is None:
            y_pred = self.minibatch_forward_pass(self.X)
        else:
            y_pred = self.minibatch_forward_pass(X)
        if training_phase:
            self.is_training = True
        return y_pred

    def validate(self, X=None):
        y_pred = self.validate_proba(X)
        return one_hot_decision_function(y_pred)

    def predict_proba(self, X):
        # split X into batches, forward pass and concat
        y_pred = self.minibatch_forward_pass(X)
        return y_pred

    @property
    def is_training(self):
        return self.training

    @is_training.setter
    def is_training(self, value):
        self.training = value
        for layer in self.layers:
            if hasattr(layer, "is_training"):
                layer.is_training = value

    def max_norm_update(self):
        for layer in self.layers:
            if hasattr(layer, "max_norm"):
                layer.max_norm_update()


def train(
    X,
    y,
    X_test,
    y_test,
    folds: int,
    layers: list,
    epochs: int,
    learning_rate: float,
    rho: float,
) -> dict:
    """
    Main function to train the model
    """
    num_classes = 4
    fold = 1
    y_test = to_categorical(y_test, num_classes)
    results = dict()

    print("Performing k-fold validation ...")

    for train, test in StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=1234
    ).split(X, y):
        X_train = X[train]
        y_train = y[train]
        y_train = to_categorical(y_train, num_classes)
        X_val = X[test]
        y_val = y[test]
        y_val = to_categorical(y_val, num_classes)

        model = NeuralNetwork(
            layers=layers,
            n_batches=1024,
            shuffle=True,
            random_seed=4321,
            optimizer_params=dict(
                max_epochs=epochs, verbose=True, learning_rate=learning_rate, rho=rho
            ),
        )

        print("\n––– Training on Fold {0} –––".format(fold))

        # Fit using this fold
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        print("Evaluating model for Fold {0} ...".format(fold))
        with Stopwatch(verbose=True):
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

        print(
            "\n–– Test accuracy for Fold {0}: {1:.2f}%\n–– Error: {2:.2f}%\n\n**************************".format(
                fold, 100.0 * score, 100.0 * (1.0 - score)
            )
        )

        # Update the results
        results.update({"Fold {0}".format(fold): [score, model]})

        # Move on to next fold
        fold += 1

    return results


def start(
    layers: list, epochs: int, learning_rate: float, rho: float, number_of_folds: int
) -> NeuralNetwork:
    """
    Starter function that returns the best trained model
    """
    print("Loading data ...")
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0

    results = train(
        X_train.copy(),
        y_train.copy(),
        X_test,
        y_test,
        folds=number_of_folds,
        layers=layers,
        epochs=epochs,
        learning_rate=learning_rate,
        rho=rho,
    )

    # Get the fold with the maximum test accuracy
    best_result = max(results, key=lambda x: results[x][0])

    print(
        "\n–– *** RESULTS *** ––\nThe best fold out of {0} is {1} with test accuracy {2:.2f}%".format(
            number_of_folds, best_result, 100 * results[best_result][0]
        )
    )

    # Return the best result
    return results[best_result][1]
