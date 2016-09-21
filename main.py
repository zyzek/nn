import network
import util


#test_set = util.load_10_batch(0)
#test_set = util.load_100_set("test")
#training_set = util.load_100_set("train")
#net = util.load_object("external/3072-500-100-10_50")
net = network.BackpropNetwork([32 * 32 * 3, 500, 100, 20])
test_set = util.load_20_set("test")
training_set = util.load_20_set("train")

net.train(training_set, batch_size=10, epochs=12, lrn_rate=0.5,
          decay_param=5.0, unfriction=0.1, eval_set=test_set,
          save_best=True, print_batches=100)


result = net.test(test_set, print_progress=True)[1]
util.save_image("20-conf-matrix.png", result, (20, 20), (800, 800))
