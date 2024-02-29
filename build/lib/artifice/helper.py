""" List of helper functions. """

import pickle
from typing import Tuple
import numpy as np

from artifice.network import NN


def apply_random_permutation(
    array_1: np.ndarray, array_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the same random permutation operation on two arrays.

    :params array_1: First array to permute.
    :params array_2: Second array to permute.
    :returns: Permuted arrays.
    """

    # Check that both arrays have same
    if len(array_1) != len(array_2):
        raise ValueError("Both arrays must be the same length.")

    # Get random permutation
    p = np.random.permutation(len(array_1))

    # Apply random permutation
    array_1 = array_1[p]
    array_2 = array_2[p]

    return array_1, array_2


def one_hot_encode(labels):
    """
    Performs One-Hot-Encoding of labels.

    :params filename:
    :returns encoded_labels:
    """

    num_unique_elements = np.unique(labels)

    dic = {}
    counter = 0
    for i in num_unique_elements:
        if i in dic:
            pass
        else:
            dic[i] = counter
            counter += 1

    encoded_labels = np.zeros((len(labels), len(num_unique_elements)))

    for index, label in enumerate(labels):
        encoded_labels[index][dic[label]] = 1.0

    return encoded_labels


def load_model(filename="Saved_Model"):
    """
    Loads a model.

    :params filename:
    """

    load_path = "Saved Models/" + filename

    with open(load_path, "rb") as fp:  # Unpickling
        loaded = pickle.load(fp)

    weights = loaded[0]
    activations = loaded[1]

    nn = NN()  # initialize a blank Neural Network
    nn.weights = weights
    nn.activation_funcs = activations

    return nn
