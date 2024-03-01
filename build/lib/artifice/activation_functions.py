""" Activation functions. """

from sympy import symbols, lambdify, Piecewise, exp, tanh

class ActivationFunction:

    def __init__(self):
        # declare the variables of interest (x for activations, y and y_hat for loss)
        self.x = symbols("x", real=True)
        self.expression = None

    def evaluate(self, x, diff = False):

        assert self.expression is not None, "Activation function is not set!"

        if diff:
            self.func = lambdify((self.x), self.expression.diff(self.x))
        else:
            self.func = lambdify((self.x), self.expression)

        return self.func(x)

class ReLU(ActivationFunction):
    def __init__(self):

        super().__init__()

        self.expression = Piecewise((0,self.x<0),(self.x, self.x>=0))

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'

class Sigmoid(ActivationFunction):
    def __init__(self):

        super().__init__()

        self.expression = 1/(1+exp(-self.x))

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'


class Tanh(ActivationFunction):
    def __init__(self):

        super().__init__()

        self.expression = tanh(self.x)

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'


class Linear(ActivationFunction):
    def __init__(self):

        super().__init__()

        self.expression = self.x

    def __repr__(self):
        return f'Relu Activation Function: \n{self.expression}'