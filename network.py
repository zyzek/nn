import sys
import numpy as np
from functools import reduce
from random import shuffle

import nn_functions as nnf
import util


class BackpropNetwork:
    def __init__(self, sizes, xfer_fn=nnf.sigmoid, cost_fn=nnf.ce_cost, reg_fn=nnf.decay_L2):
        """Initialise a backpropagation network with len(sizes) - 1 layers (input layer is implicit).
        The self.layers[i] will contain sizes[i+1] neurons.

        Args:
            sizes: A list of integers specifying the number of neurons in each layer
            xfer_fn: The transfer function to use for neurons.
            cost_fn: A measure of how well this network is approximating expected classifications.
            reg_fn: regularisation function to apply during gradient descent.
          """

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

        # Weighted input (w*x + b), and activation vectors for each layer.
        w_ins = np.array([np.empty((layer.size, 1)) for layer in self.layers])
        l_outs = np.array([np.empty((layer.size, 1)) for layer in self.layers])

        # Initialise the first hidden layer's output.
        w_ins[0] = self.layers[0].weighted_inputs(in_vec)
        l_outs[0] = self.layers[0].xfer_fn(w_ins[0])

        # Feed forward the signals, saving the layer outputs as we go.
        for i in range(1, len(self.layers)):
            w_ins[i] = self.layers[i].weighted_inputs(l_outs[i-1])
            l_outs[i] = self.layers[i].xfer_fn(w_ins[i])

        return w_ins, l_outs

    def train(self, data, batch_size, epochs, lrn_rate, decay_param, unfriction,
              eval_set=None, diminish_lrn_rate=True, save_best=True, print_batches=0):
        """Adjust the weights of this network given a training set of (input, expected_output) pairs.

        Args:
            data: (input, expected_output) pairs. Input should be the same length as the input layer.
                  Expected output should be the same length as the output layer.
            batch_size: The number of training examples to average each batch, for approximating the local gradient
                        faster than averaging over the entire training set.
            epochs: The number of times to train over the entire set before halting the training process.
            decay_param: A parameter determining how influential the regularisation decay is.
            unfriction: gradient descent momentum coefficient
            eval_set: if not None, then evaluate classification accuracy on this set after every epoch
            diminish_lrn_rate: if True, decrease the learning rate whenever accuracy goes down between epochs
            save_best: if True, save the network every time a new maximum accuracy is attained
            print_batches: over the course of an epoch, print n progress updates.
        """

        last_accuracy = 0
        best_accuracy = 0

        for i in range(epochs):
            # Split the data set up into a number of training batches.
            shuffle(data)
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

            # For each batch, estimate the gradient and descend it.
            for j, batch in enumerate(batches):
                self.gradient_descent(batch, len(data), lrn_rate, decay_param, unfriction)

                if print_batches != 0 and not ((j+1) % (len(batches)//print_batches)):
                    print("    Batch: {}/{}".format(j+1, len(batches)))

            print("Epoch {} of {} completed.".format(i+1, epochs))

            # Evaluate the classification accuracy after this epoch completes
            if eval_set is not None:
                print("Evaluating accuracy... ", end="")
                sys.stdout.flush()
                accuracy = self.test(eval_set)[0]
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
        """Estimate the gradient from a batch of example data by applying backpropagation, and descend it.

        Args:
            batch: the current batch to determine the gradient from
            input_size: the total size of the input set, used to scale the learning rate for this batch
            lrn_rate: the size of steps to take when performing gradient descent
            decay_param: network regularisation factor
            unfriction: momentum coefficient
        """

        # Begin backpropogation with the first batch.
        w_grad, b_grad = self.backprop(batch[0])

        # Sum the partial derivatives derived from doing backprop over all examples
        # in order to obtain deltas for all weights and biases.
        for example in batch[1:]:
            w_res, b_res = self.backprop(example)
            for i in range(len(w_res)):
                w_grad[i] += w_res[i]
            b_grad += b_res

        # We will multiply by an appropriate scalar to obtain the average gradient over this batch.
        batch_scale = lrn_rate/len(batch)
        for i in range(len(w_grad)):
            w_grad[i] *= batch_scale
        b_grad *= batch_scale

        # Update velocity terms and apply regularisation (which will tend to diminish weights in the network)
        decay_factor = (decay_param*lrn_rate)/input_size
        for i, layer in enumerate(self.layers):
            layer.w_vs = unfriction*layer.w_vs - w_grad[i]
            layer.ws += layer.w_vs - self.reg_fn(layer.ws, decay_factor)
            layer.b_vs = unfriction*layer.b_vs - b_grad[i]
            layer.bs += layer.b_vs

    def backprop(self, example):
        """Push a single example through the network, calculate the error of its classification,
        and propagate that error back through the network. Return the resulting gradient vectors."""
        eg_input, exp_output = example
        weighted_ins, layer_outs = self.layer_outputs(eg_input)

        # Error of the output layer:
        error = np.empty_like(layer_outs)
        error[-1] = self.cost_fn.grad(layer_outs[-1], exp_output, weighted_ins[-1])

        # Perform the backpropagation itself to produce all error terms
        for i in range(len(error) - 2, -1, -1):
            error[i] = np.dot(self.layers[i+1].ws.transpose(), error[i+1]) * self.layers[i].xfer_fn.dx(weighted_ins[i])

        w_grad = [np.empty_like(layer.ws) for layer in self.layers]
        w_grad[0] = error[0] * eg_input.transpose()
        for i in range(1, len(w_grad)):
            w_grad[i] = error[i] * layer_outs[i-1].transpose()

        return w_grad, error

    def test(self, test_set, print_progress=False):
        """Test the classification accuracy of this network over test_set.
        Return the accuracy, and a confusion matrix of the resulting classifications.

        Args:
          test_set: a sequence of (example, expected_output) pairs to test against.
          print_progress: print running classification totals.
        """
        correct = 0
        dim = len(test_set[0][1])
        results = np.zeros((dim, dim))

        for i, example in enumerate(test_set):
            result = self.evaluate(example[0])

            predicted = np.argmax(result)
            expected = np.argmax(example[1])

            results[predicted][expected] += 1
            if predicted == expected:
                correct += 1

            if print_progress and (((i+1) % (len(test_set) // 10) if len(test_set) >= 10 else 1) == 0):
                print("{}/{} correct".format(correct, i+1))

        accuracy = correct / len(test_set)
        return accuracy, results

class ConnectedLayer:
    """A fully-connected neural network layer."""
    def __init__(self, prev_size, size, xfer_fn):
        """Args:
            prev_size: number of neurons in the previous layer.
            size: number of neurons in this layer
            xfer_fn: the activation function to use for the neurons in this layer

        The number of incoming edges to this layer is size*prev_size.
        """

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
        """Given the input vector x, return the result of feeding that input through this layer."""
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = self.xfer_fn(np.dot(p[0], x) + p[1])
        return out

    def weighted_inputs(self, x):
        """Produce the result of feeding the input x thorugh this layer, without applying the activation function."""
        out = np.empty((self.size, 1))
        for i, p in enumerate(zip(self.ws, self.bs)):
            out[i] = np.dot(p[0], x) + p[1]
        return out


def softmax(sequence):
    e = np.exp(sequence)
    return e / np.sum(e)
