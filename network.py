import numpy as np
from functools import reduce
from random import shuffle

from activation_functions import sig


class BackpropNetwork:
    def __init__(self, sizes):
        """Initialise a backpropagation network with len(sizes) - 1 layers (input layer is implicit).
        The self.layers[i] will contain sizes[i+1] neurons."""

        self.layers = [ConnectedLayer(p, s, sig()) for p, s in zip(sizes[:-1], sizes[1:])]

    def evaluate(self, in_vec):
        """Feed in_vec to this network and return the output activations."""
        return reduce(lambda out_vec, layer: layer.activate(out_vec), self.layers, in_vec)

    def layer_outputs(self, in_vec):
        """Feed in_vec to this network and return the weighted inputs and activations of each layer."""
        w_ins = np.array([np.empty((layer.size, 1)) for layer in self.layers])
        l_outs = np.array([np.empty((layer.size, 1)) for layer in self.layers])

        w_ins[0] = self.layers[0].weighted_inputs(in_vec)
        l_outs[0] = self.layers[0].xfer_func(w_ins[0])
        for i in range(1, len(self.layers)):
            w_ins[i] = self.layers[i].weighted_inputs(l_outs[i-1])
            l_outs[i] = self.layers[i].xfer_func(w_ins[i])

        return w_ins, l_outs

    def train(self, data, batch_size, epochs, lrn_rate, print_batches=0):
        """Adjust the weights of this network given a training set of (input, expected_output) pairs."""

        for i in range(epochs):
            shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

            for j, batch in enumerate(batches):
                self.gradient_descent(batch, lrn_rate)

                if print_batches != 0 and not ((j+1) % (len(batches)//print_batches)):
                    print("    Batch: {}/{}".format(j+1, len(batches)))

            print("Epoch {} of {} completed.".format(i+1, epochs))


    def gradient_descent(self, data, lrn_rate):

        w_grad, b_grad = self.backprop(data[0])

        for d in data[1:]:
            w_res, b_res = self.backprop(d)
            for i in range(len(w_res)):
                w_grad[i] += w_res[i]
            b_grad += b_res

        for i in range(len(w_res)):
            w_grad[i] *= len(data)/lrn_rate
        b_grad *= len(data)/lrn_rate

        for i, layer in enumerate(self.layers):
            pre_w, pre_b = layer.ws.shape, layer.bs.shape
            layer.ws -= w_grad[i]
            layer.bs -= b_grad[i]

    def backprop(self, example):
        eg_input, exp_output = example
        weighted_ins, layer_outs = self.layer_outputs(eg_input)

        # Error of the output layer:
        # grad(cost function)
        #   element-wise multiplied with
        # (derivative of the transfer function applied to weighted inputs of last layer)
        error = np.empty_like(layer_outs)
        error[-1] = (layer_outs[-1] - exp_output) * self.layers[-1].xfer_func.dx(weighted_ins[-1])

        for i in range(len(error) - 2, -1, -1):
            error[i] = np.dot(self.layers[i+1].ws.transpose(), error[i+1]) * self.layers[i].xfer_func.dx(weighted_ins[i])

        w_grad = [np.empty_like(layer.ws) for layer in self.layers]
        w_grad[0] = error[0] * eg_input.transpose()
        for i in range(1, len(w_grad)):
            w_grad[i] = error[i] * layer_outs[i-1].transpose()

        return w_grad, error


class ConnectedLayer:
    def __init__(self, prev_size, size, xfer_func):
        self.prev_size = prev_size
        self.size = size
        self.ws = np.random.randn(size, prev_size)
        self.bs = np.random.randn(size, 1)
        """Weights. The last element of each set of weights is a bias."""

        self.xfer_func = xfer_func

    def activate(self, x):
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = self.xfer_func(np.dot(p[0], x) + p[1])
        return out
        #return np.array([self.xfer_func(np.dot(w, x) + b) for w, b in zip(self.ws, self.bs)])

    def weighted_inputs(self, x):
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = np.dot(p[0], x) + p[1]
        return out
        #return np.array([np.dot(w, x) + b for w, b in zip(self.ws, self.bs)])

def evaluate_network(net, test_set):
    correct = 0
    results = [0]*len(test_set[0][1])

    for i, example in enumerate(test_set):
        result = net.evaluate(example[0])

        predicted = np.argmax(result)
        expected = np.argmax(example[1])

        results[predicted] += 1
        if predicted == expected:
            correct += 1

        if ((i+1) % (len(test_set) // 10)) == 0:
            print("{}/{} correct".format(correct, i+1))

    print(result)
    print("{:.2f}% accuracy".format(100.0 * correct / len(test_set)))