import numpy as np

class sigmoid:
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dx(self, x):
        s = self(x)
        return s * (1.0 - s)

class CostFunction:
    def __init__(self, xfer_fn):
        self.xfer_fn = xfer_fn

class quad_cost(CostFunction):
    def __call__(self, x, y):
        """MSE of the vector x relative to y."""
        return 0.5 * np.sum((x - y)**2)

    def grad(self, x, y, z):
        """Vector of partial derivative errors."""
        return (x - y) * self.xfer_fn.dx(z)

class ce_cost(CostFunction):
    def __call__(self, x, y):
        return np.sum((y - 1.0)*np.log(1 - x) - y*np.log(x))

    def grad(self, x, y, z):
        """Vector of partial derivative errors."""
        return x - y