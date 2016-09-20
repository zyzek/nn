import network
import util

test_set = util.load_10_batch(0, normalise=True)
net = network.BackpropNetwork([32 * 32 * 3, 100, 10])

for i in range(1, 6):
    net.train(util.load_10_batch(i), batch_size=10, epochs=1, lrn_rate=0.2*(0.3**i), \
              decay_param=5.0, unfriction=0.1, print_batches=10)
    accuracy = network.evaluate_network(net, test_set)
    util.save_object(net, "3072-100-10_{}".format(int(100 * accuracy)))

network.evaluate_network(net, test_set, print_progress=True)
