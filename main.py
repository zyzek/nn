import network
import util

test_set = util.load_10_batch(0)
net = network.BackpropNetwork([32 * 32 * 3, 100, 50, 30, 10])

for i in range(1, 6):
    net.train(util.load_10_batch(i), batch_size=10, epochs=6, lrn_rate=0.5*(0.1**(i//3)), print_batches=10)
    accuracy = network.evaluate_network(net, test_set)
    util.save_object(net.export_weights(), "3072-100-50-30-10_{}".format(int(100 * accuracy)))

util.save_object(net.export_weights(), "3072-100-50-30-10")

network.evaluate_network(net, test_set, print_progress=True)
