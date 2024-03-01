""" Loss functions. """

from sympy import symbols, lambdify, Abs, log

class LossFunction:

    def __init__(self):
        
        # declare the variables of interest (x for activations, y and y_hat for loss)
        self.y_pred, self.y_true = symbols("y_pred y_true", real=True)
        self.expression = None

    def evaluate(self, y_pred, y_true, diff = False):

        assert self.expression is not None, "Loss function is not set!"

        if diff:
            self.func = lambdify((self.y_pred, self.y_true), self.expression.diff(self.y_pred))
        else:
            self.func = lambdify((self.y_pred, self.y_true), self.expression)

        return self.func(y_pred, y_true)


class MeanSquaredError(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum((self.y_pred - self.y_true)**2)


    def __repr__(self):
        return f'Mean Squared Error Loss Function: \n{self.expression}'

class MeanAbsoluteError(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum(Abs(self.y_pred - self.y_true))

class MeanAbsolutePercentError(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum(100 * Abs((self.y_pred - self.y_true) / (self.y_true)))

class MeanLogSquaredError(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum((log(self.y_pred + 1.) - log(self.y_true + 1.))**2)

class BinaryCrossEntropy(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum(-(self.y_true*log(self.y_pred) + (1 - self.y_true)*log(1 - self.y_pred)))

class PoissonError(LossFunction):

    def __init__(self):
        super().__init__()

        self.expression = sum(self.y_pred - self.y_true * log(self.y_pred))