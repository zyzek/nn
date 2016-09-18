import network
import numpy as np
import imageprocessing as ip

#training_set = ip.load_10_batch(1)
#test_set = ip.load_10_batch(0)
#net = network.BackpropNetwork([32 * 32 * 3, 100, 50, 30, 10])
#net.train(training_set, batch_size=10, epochs=3, lrn_rate=0.1, print_batches=10)

training_set, test_set = ip.load_mnist()
net = network.BackpropNetwork([28*28, 100, 50, 10])
net.train(training_set, batch_size=1000, epochs=1, lrn_rate=0.1, print_batches=10)
network.evaluate_network(net, test_set)
