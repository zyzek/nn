import network
import util
import random

test_set = util.load_10_batch(0, normalise=True)
#net = network.BackpropNetwork([32 * 32 * 3, 500, 100, 10])
net = util.load_object("3072-500-100-10_34")

eval_size = 1000

for i in range(1, 6):
    test_index = random.randint(0, len(test_set) - eval_size)
    net.train(util.load_10_batch(i), batch_size=10, epochs=12, lrn_rate=0.1*(0.7**i), \
              decay_param=5.0, unfriction=0.1, eval_set=test_set[test_index:test_index+1000], \
              save_best=True, print_batches=10)

net.evaluate_network(test_set, print_progress=True)
