import numpy as np

class sig:
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dx(self, x):
        s = self(x)
        return s * (1.0 - s)