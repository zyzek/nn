import network
import util

# TODO: tanh, rectified linear neurons
# TODO: hyperparameter search
# TODO: auto learning rate scheduling
# TODO: convolutional
# TODO: vectorisation (particularly of gradient_descent())


test_set = util.load_10_batch(0)
net = network.BackpropNetwork([32 * 32 * 3, 100, 50, 30, 10])

for i in range(1, 6):
    net.train(util.load_10_batch(i), batch_size=10, epochs=6, lrn_rate=0.5*(0.1**(i//3)), \
              decay_param=5.0, unfriction=0.1, print_batches=10)
    accuracy = network.evaluate_network(net, test_set)
    util.save_object(net, "3072-100-50-30-10_{}".format(int(100 * accuracy)))

network.evaluate_network(net, test_set, print_progress=True)
