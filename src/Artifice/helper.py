""" List of helper functions. """

import pickle
import numpy as np

from artifice.artifice import NN


def unison_shuffled_copies(a, b):
    """
    ???.

    :params a:
    :params b:
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


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
