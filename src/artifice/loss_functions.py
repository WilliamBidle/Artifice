# pylint: disable=too-few-public-methods

""" Loss functions. """

import numpy as np
from sympy import symbols, lambdify, Abs, log

# Initialize symbols to be used for function generation
Y_PRED, Y_TRUE = symbols("y_pred y_true", real=True)


class LossFunction:
    """ Loss function base-class. """

    def __init__(self):

        # Try to initialize the derivitive of the loss function
        try:
            self.expression_diff = self.expression.diff(Y_PRED)
        except AttributeError:
            self.expression = None
            self.expression_diff = None

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray, diff: bool = False):
        """ 
        Evaluate the loss function given the true and predicted values. 

        :param y_pred: The predicted values of the labels.
        :param y_true: The truth values of the labels.
        :param diff: Whether or not to evalutate the derivitive of the loss function.
        :returns: Evaluation of the loss function or it's derivitive.
        """

        assert self.expression is not None, "Loss function is not set!"

        # Check for differentiation and prepare the function for evaluation
        if diff:
            func = lambdify((Y_PRED, Y_TRUE), self.expression_diff)
        else:
            func = lambdify((Y_PRED, Y_TRUE), self.expression)

        # Need to return the sum of the loss function if not differentiating
        return func(y_pred, y_true) if diff else sum(func(y_pred, y_true))


class MeanSquaredError(LossFunction):
    """ MeanSquaredError loss function. """

    def __init__(self):

        self.expression = (Y_PRED - Y_TRUE)**2

        super().__init__()

    def __repr__(self):

        return f'Mean Squared Error Loss Function: \n{self.expression}'


class MeanAbsoluteError(LossFunction):
    """ MeanAbsoluteError loss function. """

    def __init__(self):

        self.expression = Abs(Y_PRED - Y_TRUE)

        super().__init__()

    def __repr__(self):

        return f'Mean Absolute Error Loss Function: \n{self.expression}'


class MeanAbsolutePercentError(LossFunction):
    """ MeanAbsolutePercentError loss function. """

    def __init__(self):

        self.expression = 100 * Abs((Y_PRED - Y_TRUE) / (Y_TRUE))

        super().__init__()

    def __repr__(self):

        return f'Mean Absolute Percent Error Loss Function: \n{self.expression}'


class MeanLogSquaredError(LossFunction):
    """ MeanLogSquaredError loss function. """

    def __init__(self):

        self.expression = (log(Y_PRED + 1.) - log(Y_TRUE + 1.))**2

        super().__init__()

    def __repr__(self):

        return f'Mean Logarithmic Squared Error Loss Function: \n{self.expression}'


class BinaryCrossEntropy(LossFunction):
    """ BinaryCrossEntropy loss function. """

    def __init__(self):

        self.expression = -(Y_TRUE*log(Y_PRED) + (1 - Y_TRUE)*log(1 - Y_PRED))

        super().__init__()

    def __repr__(self):

        return f'Binary Cross-Entropy Loss Function: \n{self.expression}'


class PoissonError(LossFunction):
    """ PoissonError loss function. """

    def __init__(self):

        self.expression = Y_PRED - Y_TRUE * log(Y_PRED)

        super().__init__()

    def __repr__(self):

        return f'Poisson Loss Function: \n{self.expression}'
