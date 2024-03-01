""" Code written by William Bidle and Ilana Zane """

__version__ = "dev"

import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from artifice.loss_functions import (
    MeanAbsoluteError,
    MeanAbsolutePercentError,
    MeanLogSquaredError,
    MeanSquaredError,
    PoissonError,
    BinaryCrossEntropy,
)
from artifice.activation_functions import (
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
)


class NN:

    """
    Defines a neural network.

    :param layer_sequence: A list containing the nodes per layer and correcponding activation
    functions between layers.
    :param loss_function: The desired loss function to be used.
    """

    def __init__(self, layer_sequence: Optional[list] = None, loss_function: Optional[str] = None):

        valid_loss_functions = {
            "MSE": MeanSquaredError(),
            "MAE": MeanAbsoluteError(),
            "MAPE": MeanAbsolutePercentError(),
            "MLSE": MeanLogSquaredError(),
            "Poisson": PoissonError(),
            "BCE": BinaryCrossEntropy(),
        }

        valid_activation_functions = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "linear": Linear(),
        }

        if layer_sequence is not None:

            # Separate out the layer information and activations (every other element)
            layers = layer_sequence[::2]
            activation_funcs = layer_sequence[1::2]

            # Check if valid layer elements activation functions
            assert all(isinstance(item, int)
                       for item in layers), "Invalid Layer Sequence!"
            for item in activation_funcs:
                assert item in valid_activation_functions, f"Invalid activation function '{
                    item}'."

            # Set the layers
            layers = np.array(layer_sequence[::2], dtype=int)

            # initialize the weights based off of the desired layer sequence
            self.weights = self.__initialize_weights(layers)

            # initialize the 'activation_funcs' property
            self.activation_funcs = []

            # initialize each declared activation functions between layers
            for activation_func in activation_funcs:
                self.activation_funcs.append(
                    valid_activation_functions[activation_func])

            # initialize the loss_func
            self.loss_func = valid_loss_functions[loss_function]

            # initialize the loss_func_label (used in plotting for now)
            self.loss_func_label = loss_function

        else:
            self.weights = None
            self.activation_funcs = None
            self.loss_func = None
            self.loss_func_label = None

        # initialize the *training_err* property (will be set later once the model is trained)
        self.training_err = None

    def __initialize_weights(self, layers: np.ndarray) -> List[np.ndarray]:
        """
        Initialize the weights of the network.

        :params layers: An array containing the layer information of the network.
        :returns weights: List containing the 2D weight arrays between the different layers.
        """

        layers_reorganized = np.flip(
            layers.repeat(2)[1:-1].reshape(len(layers) - 1, 2), axis=1
        )

        # initialize the list of the weights between different layers
        weights = []

        for layer_reorganized in layers_reorganized:
            # include bias vector with the '+ 1'
            weight = np.random.randn(
                layer_reorganized[0], layer_reorganized[1] + 1)

            # HE initialization for weights
            weights.append(weight * np.sqrt(2 / layer_reorganized[1]))

        return weights

    def __update_weights(
        self, weights: List[np.ndarray], layer_values: List[List], _label_: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Update the weights of the network.

        :params weights: List containing the 2D weight arrays between the different layers.
        :params layer_values: List containing the values of each layer for a given input value.
        :params _label_: Array representation for the current label, usually one hot ecoded.
        :returns weight_updates: List of updated 2D weight arrays between the different layers.
        :returns weights: List of original 2D weight arrays between the different layers.
        """

        # get the list of desired activation functions
        activations = self.activation_funcs

        # make a copy of the weights so they aren't changed
        weight_updates = weights.copy()

        # blue in notes
        blue = np.diag(self.loss_func.evaluate(
            layer_values[-1], _label_, diff=True))

        # need to add an extra component to input for bias
        layer_output = np.dot(
            weights[-1], np.concatenate((layer_values[-2], [1])))

        # red in notes
        red = activations[-1].evaluate(layer_output, diff=True)

        # index through each weight (work backwards)
        for i in range(len(weights), 0, -1):

            # pink in notes
            pink = np.concatenate((layer_values[i - 1], [1]))

            # first two terms in gradient
            grad = np.matmul(blue, np.outer(red, pink))

            # look forwards through each weight (only if there are forward weights)
            for j in range(len(weights), i, -1):

                # orange in notes
                orange = np.transpose(weights[j - 1])

                # green in notes
                green = np.diag(
                    activations[j - 1].evaluate(
                        np.dot(
                            weights[j - 2],
                            np.concatenate((layer_values[j - 2], [1])),
                        ),
                        diff=True,
                    )
                )

                # add on the bias vector to make sure dimensions work properly
                bias_vec = np.ones((len(green), 1))

                # incorperate the bias
                green = np.hstack((green, bias_vec))

                # now multiply the rest to grad
                grad = np.matmul(green, np.matmul(orange, grad))

            # record the change in weight
            weight_updates[i - 1] = grad

        return weight_updates, weights

    def get_network_outputs(
        self, weights: List[np.ndarray], _input_: np.ndarray
    ) -> np.ndarray:
        """
        Initialize the weights of the network.

        :params weights: List containing the 2D weight arrays between the different layers.
        :params _inputs_: Input layer to the network.
        :returns network_outputs: Output layer of the network.
        """

        # get the list of desired activation functions
        activations = self.activation_funcs

        # add the first layer to the list
        current_layer = _input_
        network_outputs = [current_layer]

        for index, weight in enumerate(weights):

            # need to add an extra component to input for bias
            layer_output = np.dot(weight, np.concatenate((current_layer, [1])))

            current_layer = activations[index].evaluate(
                layer_output, diff=False)

            network_outputs.append(current_layer)

        return network_outputs

    def compute_error(self, _result_: np.ndarray, _label_: np.ndarray) -> float:
        """
        Compute the error between the neural network's output and expected value.

        :params _result_: Output layer of the network.
        :params _label_: Expected result.
        :returns error: The error of the network.
        """
        error = self.loss_func.evaluate(_result_, _label_, diff=False)

        return error

    def train(  # pylint: disable=too-many-arguments, too-many-locals
        self, x_train, y_train, batch_size=1, epochs=1, epsilon=1, visualize=False
    ) -> None:
        """
        Train a model.

        :params x_train:
        :params y_train:
        :params batch_size:
        :params epochs:
        :params epsilon:
        :params visualize:
        """

        assert self.weights is not None, "Cannot train because weights have not been initialized!"

        weights = self.weights  # get the list of weights

        error_list = []

        counter = 0  # keep track of the current iteration

        weights_list = (
            {}
            # create a dictionary to keep track of the weight updates (batch size)
        )

        # just a temporary blank array since training hasn't begun yet
        for i in range(len(weights)):
            weights_list[i] = []

        for i in range(epochs):

            # iterate through the inputs and labels
            for _input_, _label_ in tqdm(
                zip(x_train, y_train), total=len(x_train), desc=f"Epoch {str(i + 1)}"
            ):

                network_output = self.get_network_outputs(
                    weights, _input_
                )  # the current network output

                error = self.compute_error(network_output[-1], _label_)

                weight_updates, weights = self.__update_weights(
                    weights, network_output, _label_
                )

                for j in range(len(weights)):
                    weights_list[j].append(weight_updates[j])

                counter += 1

                if (counter) % batch_size == 0:
                    for index, weight in enumerate(weights):

                        weights[index] = weight - epsilon * np.average(
                            np.array(weights_list[index]), axis=0
                        )
                        weights_list[index] = []

                error_list.append(error)

            self.weights = weights
            self.training_err = error_list

        if visualize is True:
            _, ax = plt.subplots(figsize=(12, 6))

            ax.plot(self.training_err)  # to visualize error over time

            ax.set_xlabel("Training Sample", fontsize=14)
            ax.set_ylabel(f"{self.loss_func_label} Error", fontsize=14)

            ax.grid(linestyle="--")

            plt.show()

    def evaluate(self, x_test):
        """
        Evaluate a model.

        :params x_test:
        :returns results:
        """

        results = []

        for _input_ in tqdm(x_test, desc="Evaluating Test Data", total=len(x_test)):
            # the current network output
            network_output = self.get_network_outputs(self.weights, _input_)
            results.append(network_output[-1])

        return results

    def save_model(self, out_dir: str, filename: str) -> None:
        """
        Save a model.

        :params filename:
        """

        # Enforce trailing backslash to directory
        out_dir = os.path.join(out_dir, "")

        # Check if out_dir exists
        if not os.path.exists(out_dir):
            raise ValueError(f"Invalid path: {out_dir}.")

        save_path = out_dir + filename

        to_save = [
            self.weights,
            self.activation_funcs,
        ]  # save both the activations and weights
        with open(save_path, "wb") as fp:  # save the weights and activations
            pickle.dump(to_save, fp)

        print()
        print(f"Model saved at '{save_path}'")
        print()
