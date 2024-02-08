""" Tests for the artifice.helper module. """

import numpy as np

from artifice.helper import apply_random_permutation, one_hot_encode


def test_apply_random_permutation():
    """Testing the apply_random_permutation function."""

    arr_1, arr_2 = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])
    permuted_arr_1, permuted_arr_2 = apply_random_permutation(arr_1, arr_2)

    # Check that the permuted arrays permuted the same
    assert np.allclose(permuted_arr_1, permuted_arr_2)


def test_one_hot_encode():
    """Testing the one_hot_encode function."""

    # Check that unique elements are encoded correctly
    arr = np.array([1, 2, 3])
    one_hot_encoded_arr = one_hot_encode(arr)
    assert np.allclose(one_hot_encoded_arr, np.identity(3))

    # Check that all similar elements are encoded correctly
    arr = np.array([1, 1, 1])
    one_hot_encoded_arr = one_hot_encode(arr)
    assert np.allclose(one_hot_encoded_arr, np.ones(shape=(3, 1)))
