import numpy as np


class sigmoid:
    """Logistic/sigmoid activation function."""
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dx(self, x):
        s = self(x)
        return s * (1.0 - s)


# Neural network cost functions.

class CostFunction:
    def __init__(self, xfer_fn):
        self.xfer_fn = xfer_fn


class quad_cost(CostFunction):
    """Mean squared error / quadratic cost function."""
    def __call__(self, x, y):
        """MSE of the vector x relative to y."""
        return 0.5 * np.sum((x - y)**2)

    def grad(self, x, y, z):
        """Vector of partial derivative errors."""
        return (x - y) * self.xfer_fn.dx(z)


class ce_cost(CostFunction):
    """Cross-entropy cost function."""
    def __call__(self, x, y):
        return np.sum((y - 1.0)*np.log(1 - x) - y*np.log(x))

    def grad(self, x, y, z):
        """Vector of partial derivative errors."""
        return x - y


# Regularisation decay functions.

def decay_L2(weight, scale):
    return scale * weight


def decay_L1(weight, scale):
    return scale * np.sign(weight)


def decay_none(weight, scale):
    return 0
