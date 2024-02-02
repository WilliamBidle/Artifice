""" Code written by William Bidle and Ilana Zane """

__version__ = "dev"

import os
import json
import pickle
from typing import List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import sympy
from tqdm import tqdm


class NN:

    """
    Defines a neural network.

    :param layer_sequence: A list containing the nodes per layer and correcponding activation
    functions between layers.
    :param loss_function: The desired loss function to be used.
    """

    def __init__(self, layer_sequence: list = None, loss_function: str = "MSE"):

        # Load in the function library
        self.activation_funcs_library = self.__load_func_libraries(
            "activation_funcs_library.txt"
        )

        # Separate out the layer information (every other element)
        layers = layer_sequence[::2]

        # check that the declared number of nodes is an integer
        if all(isinstance(item, int) for item in layers) is True:
            # set the 'layers' property
            layers = np.array(layer_sequence[::2], dtype=int)

            # initialize the *weights* property based off of the desired layer sequence
            self.weights = self.__initialize_weights(layers)

        else:
            # raise an exception if the input layer sequence is improperly defined
            raise ValueError("Invalid Layer Sequence!")

        # separate out the activation function information
        activation_funcs = layer_sequence[1::2]

        # check that the activation functions are strings
        if all(isinstance(item, str) for item in activation_funcs) is True:
            # initialize the 'activation_funcs' property
            self.activation_funcs = []

            # initialize each declared activation functions between layers
            for activation_func in activation_funcs:
                self.activation_funcs.append(
                    self.__init_func(self.activation_funcs_library, activation_func)
                )

        else:
            # raise an exception if the input layer sequence is improperly defined
            raise ValueError("Invalid Layer Sequence!")

        # initialize the *loss_funcs_library* property
        self.loss_funcs_library = self.__load_func_libraries("loss_funcs_library.txt")

        # initialize the *loss_func* property
        self.loss_func = self.__init_func(self.loss_funcs_library, loss_function)

        # initialize the *loss_func_label* property (used in plotting for now)
        self.loss_func_label = loss_function

        # initialize the *training_err* property (will be set later once the model is trained)
        self.training_err = None

    def __load_func_libraries(self, func_file: str) -> dict:

        """
        Loads in dictionaries of available functions.

        :param func_file: the filename containing the library of usable functions.
        :returns func_library: a dictionary of the usable functions.
        """

        # open the desired file
        with open(
            os.path.join(os.path.dirname(__file__), func_file), encoding="utf-8"
        ) as f:
            data = f.read()

        # reconstruct the data as a dictionary
        func_library = json.loads(data)

        return func_library

    def __init_func(
        self, func_library: dict, func_name: str
    ) -> sympy.core.symbol.Symbol:

        """
        Initialize a function from a function library

        :param func_library: A dictionary of the usable functions.
        :param func_name: The name of the mathematical function to be initialized (e.g., 'sigmoid').
        :returns expression: Symbolic mathematical representation of 'func_name'.
        """

        # try to initialize the function
        try:
            expression = func_library[func_name]

        # if the function doesn't exist within the function library, return an exception
        except Exception as exc:
            raise ValueError(
                f"Desired function '{func_name}' does not exist within the 'func_library.'"
            ) from exc

        # declare the variables of interest (x for activations, y and y_hat for loss)
        x, y, y_hat = sympy.symbols("x y y_hat", real=True)

        # parse throught the expression
        expression = sympy.parse_expr(
            expression, local_dict={"x": x, "y": y, "y_hat": y_hat}
        )

        return expression

    def __eval_func(
        self, expression: sympy.core.symbol.Symbol, vals: List[List], diff: bool = False
    ) -> Union[float, np.ndarray]:

        """
        Initialize an activation function.

        :param expression: Symbolic mathematical representation of 'func'.
        :param vals: The values to evaluate 'expression' with - 1 sub-list for activation functions
        (x information), 2 sub-lists for loss functions (y, y_hat information).
        :param diff: Whether or not to evaluate the derivitive of 'expression' at 'vals'.
        :returns result: evaluation of 'expression' at '_input_' - if diff = False -> Float,
        if diff = True -> np.ndarray.
        """

        # Evaluate Activation Functions
        if expression in self.activation_funcs:

            # the variable of interest
            x = sympy.Symbol("x", real=True)

            # differentiate only if the 'diff' flag is True
            if diff is True:
                expression = expression.diff(x)

            # allow the function to be evaluated from lists
            func = sympy.lambdify(x, expression)

            # evaluate the function at the given input
            result = func(vals[0])

        # Evaluate Loss Functions
        else:

            # the variables of interest
            y, y_hat = sympy.symbols("y, y_hat", real=True)

            # differentiate only if the 'diff' flag is True
            if diff is True:
                expression = expression.diff(y)

                # allow the function to be evaluated from lists
                func = sympy.lambdify((y, y_hat), expression)

                # evaluate the function at the given input
                result = func(vals[0], vals[1])

            else:

                # allow the function to be evaluated from lists
                func = sympy.lambdify((y, y_hat), expression)

                # evaluate the function at the given input
                result = sum(func(vals[0], vals[1]))

        return result

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
            weight = np.random.randn(layer_reorganized[0], layer_reorganized[1] + 1)

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
        blue = np.diag(
            self.__eval_func(self.loss_func, [layer_values[-1], _label_], diff=True)
        )

        # need to add an extra component to input for bias
        layer_output = np.dot(weights[-1], np.concatenate((layer_values[-2], [1])))

        # red in notes
        red = self.__eval_func(activations[-1], [layer_output], diff=True)

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
                    self.__eval_func(
                        activations[j - 1],
                        [
                            np.dot(
                                weights[j - 2],
                                np.concatenate((layer_values[j - 2], [1])),
                            )
                        ],
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

            current_layer = self.__eval_func(
                activations[index], [layer_output], diff=False
            )

            network_outputs.append(current_layer)

        return network_outputs

    def compute_error(self, _result_: np.ndarray, _label_: np.ndarray) -> float:
        """
        Compute the error between the neural network's output and expected value.

        :params _result_: Output layer of the network.
        :params _label_: Expected result.
        :returns error: The error of the network.
        """
        error = self.__eval_func(self.loss_func, [_result_, _label_])

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

        weights = self.weights  # get the list of weights

        error_list = []

        counter = 0  # keep track of the current iteration

        weights_list = (
            {}
        )  # create a dictionary to keep track of the weight updates (batch size)

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
