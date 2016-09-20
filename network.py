import sys
import numpy as np
from functools import reduce
from random import shuffle

import nn_functions as nnf
import util


class BackpropNetwork:
    def __init__(self, sizes, xfer_fn=nnf.sigmoid, cost_fn=nnf.ce_cost, reg_fn=nnf.decay_L2):
        """Initialise a backpropagation network with len(sizes) - 1 layers (input layer is implicit).
        The self.layers[i] will contain sizes[i+1] neurons."""

        self.layers = [ConnectedLayer(p, s, xfer_fn()) for p, s in zip(sizes[:-1], sizes[1:])]
        self.cost_fn = cost_fn(self.layers[-1].xfer_fn)
        self.reg_fn = reg_fn

    def export_weights(self):
        return [(layer.ws, layer.bs) for layer in self.layers]

    def import_weights(self, weights):
        for w, layer in zip(weights, self.layers):
            layer.ws = w[0]
            layer.bs = w[1]

    def evaluate(self, in_vec):
        """Feed in_vec to this network and return the output activations."""
        return reduce(lambda out_vec, layer: layer.activate(out_vec), self.layers, in_vec)

    def layer_outputs(self, in_vec):
        """Feed in_vec to this network and return the weighted inputs and activations of each layer."""
        w_ins = np.array([np.empty((layer.size, 1)) for layer in self.layers])
        l_outs = np.array([np.empty((layer.size, 1)) for layer in self.layers])

        w_ins[0] = self.layers[0].weighted_inputs(in_vec)
        l_outs[0] = self.layers[0].xfer_fn(w_ins[0])
        for i in range(1, len(self.layers)):
            w_ins[i] = self.layers[i].weighted_inputs(l_outs[i-1])
            l_outs[i] = self.layers[i].xfer_fn(w_ins[i])

        return w_ins, l_outs

    def train(self, data, batch_size, epochs, lrn_rate, decay_param, unfriction,
              eval_set=None, diminish_lrn_rate=True, save_best=True, print_batches=0):
        """Adjust the weights of this network given a training set of (input, expected_output) pairs."""
        last_accuracy = 0
        best_accuracy = 0

        for i in range(epochs):
            shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

            for j, batch in enumerate(batches):
                self.gradient_descent(batch, len(data), lrn_rate, decay_param, unfriction)

                if print_batches != 0 and not ((j+1) % (len(batches)//print_batches)):
                    print("    Batch: {}/{}".format(j+1, len(batches)))

            print("Epoch {} of {} completed.".format(i+1, epochs))

            if eval_set is not None:
                print("Evaluating accuracy... ", end="")
                sys.stdout.flush()
                accuracy = self.test(eval_set)
                print("{:.2f}%".format(100.0 * accuracy))
                
                if accuracy > best_accuracy and save_best:
                    best_accuracy = accuracy
                    name = "-".join([str(self.layers[0].prev_size)] + [str(layer.size) for layer in self.layers])
                    util.save_object(self, name + "_{}".format(int(100 * accuracy)))

                if accuracy < last_accuracy and diminish_lrn_rate:
                    print("Reducing learning rate from {} to {}.".format(lrn_rate, lrn_rate/2))
                    lrn_rate /= 2
                last_accuracy = accuracy


    def gradient_descent(self, batch, input_size, lrn_rate, decay_param, unfriction):

        w_grad, b_grad = self.backprop(batch[0])

        for example in batch[1:]:
            w_res, b_res = self.backprop(example)
            for i in range(len(w_res)):
                w_grad[i] += w_res[i]
            b_grad += b_res

        batch_scale = lrn_rate/len(batch)
        decay_factor = (decay_param*lrn_rate)/input_size

        for i in range(len(w_grad)):
            w_grad[i] *= batch_scale
        b_grad *= batch_scale

        for i, layer in enumerate(self.layers):
            layer.w_vs = unfriction*layer.w_vs - w_grad[i]
            layer.ws += layer.w_vs - self.reg_fn(layer.ws, decay_factor)
            layer.b_vs = unfriction*layer.b_vs - b_grad[i]
            layer.bs += layer.b_vs

    def backprop(self, example):
        eg_input, exp_output = example
        weighted_ins, layer_outs = self.layer_outputs(eg_input)

        # Error of the output layer:
        error = np.empty_like(layer_outs)
        error[-1] = self.cost_fn.grad(layer_outs[-1], exp_output, weighted_ins[-1])

        for i in range(len(error) - 2, -1, -1):
            error[i] = np.dot(self.layers[i+1].ws.transpose(), error[i+1]) * self.layers[i].xfer_fn.dx(weighted_ins[i])

        w_grad = [np.empty_like(layer.ws) for layer in self.layers]
        w_grad[0] = error[0] * eg_input.transpose()
        for i in range(1, len(w_grad)):
            w_grad[i] = error[i] * layer_outs[i-1].transpose()

        return w_grad, error

    def test(self, test_set, print_progress=False):
        correct = 0
        results = [0]*len(test_set[0][1])

        for i, example in enumerate(test_set):
            result = self.evaluate(example[0])

            predicted = np.argmax(result)
            expected = np.argmax(example[1])

            results[predicted] += 1
            if predicted == expected:
                correct += 1

            if print_progress and (((i+1) % (len(test_set) // 10) if len(test_set) >= 10 else 1) == 0):
                print("{}/{} correct".format(correct, i+1))

        accuracy = correct / len(test_set)
        #print("{:.2f}% accuracy".format(100.0 * accuracy))
        return accuracy

class ConnectedLayer:
    def __init__(self, prev_size, size, xfer_fn):

        # Layer size. Number of inputs is prev_size*size. Number of neurons is size.
        self.prev_size = prev_size
        self.size = size

        # Weights and biases
        self.ws = np.random.randn(size, prev_size) / np.sqrt(prev_size)
        self.bs = np.random.randn(size, 1)

        # Velocity terms for momentum
        self.w_vs = np.zeros_like(self.ws)
        self.b_vs = np.zeros_like(self.bs)

        # Transfer function for neuron activation
        self.xfer_fn = xfer_fn

    def activate(self, x):
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = self.xfer_fn(np.dot(p[0], x) + p[1])
        return out

    def weighted_inputs(self, x):
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = np.dot(p[0], x) + p[1]
        return out


