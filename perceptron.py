import numpy as np
from functools import reduce
from random import shuffle

from activation_functions import sigmoid


class PerceptronNetwork:
    def __init__(self, sizes):
        self.layers = [ConnectedLayer(p, s, sigmoid) for p, s in zip(sizes[:-1], sizes[1:])]

    def activate(self, in_vec):
        return reduce(lambda out_vec, layer: layer.activate(out_vec), self.layers, in_vec)

    def train(self, data, batch_size, epochs, learning_rate):
        for _ in range(epochs):
            shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

            for batch in batches:
                self.gradient_descent(batch, learning_rate)

    def gradient_descent(self, data, learning_rate):
        pass



class ConnectedLayer:
    def __init__(self, prev_layer_size, size, xfer_func):
        self.w = np.random.randn(size, prev_layer_size)
        """Weights"""

        self.b = np.random.rand(size)
        """Bias"""

        self.xfer_func = xfer_func

    def activate(self, x):
        return np.array([self.xfer_func(np.dot(w, x) + b) for w, b in zip(self.w, self.b)])


net = PerceptronNetwork([3, 3, 3])
print(net.activate(np.array([1, 2, 3])))