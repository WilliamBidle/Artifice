# pylint: disable=too-few-public-methods

""" Activation functions. """

import numpy as np
from sympy import symbols, lambdify, Piecewise, exp, tanh

# Initialize symbol to be used for function generation
X_SYMBOL = symbols("x", real=True)


class ActivationFunction:
    """ Loss function base-class. """

    def __init__(self):

        # Try to initialize the derivitive of the activation function
        try:
            self.expression_diff = self.expression.diff(X_SYMBOL)
        except AttributeError:
            self.expression = None
            self.expression_diff = None

    def evaluate(self, x: np.ndarray, diff=False):
        """ 
        Evaluate the activation function given the input value. 

        :param x: The input values to apply the activation function on.
        :param diff: Whether or not to evalutate the derivitive of the loss function.
        :returns: Evaluation of the activation function or it's derivitive.
        """

        assert self.expression is not None, "Activation function is not set!"

        if diff:
            func = lambdify((X_SYMBOL), self.expression.diff(X_SYMBOL))
        else:
            func = lambdify((X_SYMBOL), self.expression)

        return func(x)


class ReLU(ActivationFunction):
    """ ReLU activation function. """

    def __init__(self):

        self.expression = Piecewise(
            (0, X_SYMBOL < 0), (X_SYMBOL, X_SYMBOL >= 0))

        super().__init__()

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'


class Sigmoid(ActivationFunction):
    """ Sigmoid activation function. """

    def __init__(self):

        self.expression = 1/(1+exp(0 - X_SYMBOL))

        super().__init__()

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'


class Tanh(ActivationFunction):
    """ Tanh activation function. """

    def __init__(self):

        self.expression = tanh(X_SYMBOL)

        super().__init__()

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'


class Linear(ActivationFunction):
    """ Linear activation function. """

    def __init__(self):

        self.expression = X_SYMBOL

        super().__init__()

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'
