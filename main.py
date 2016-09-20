import network
import util
import random

test_set = util.load_10_batch(0, normalise=True)
print(len(test_set))
exit()
#net = network.BackpropNetwork([32 * 32 * 3, 500, 100, 10])
net = util.load_object("3072-500-100-10_34")

eval_size = 1000

for i in range(1, 6):
    net.train(util.load_10_batch(i), batch_size=10, epochs=1, lrn_rate=0.1*(0.7**i), \
              decay_param=5.0, unfriction=0.1, print_batches=10)
    test_index = random.randint(0, len(test_set) - eval_size)
    accuracy = network.evaluate_network(net, test_set[test_index:test_index+100])
    util.save_object(net, "3072-500-100-10_{}".format(int(100 * accuracy)))

network.evaluate_network(net, test_set, print_progress=True)
