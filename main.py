import sys
import util

"""
# Example Training and Evaluation of a network on the CIFAR-10 set.
import network

test_set = util.load_10_batch(0)
train_set = util.load_10_batch(1)

net = network.BackpropNetwork([32*32*3, 100, 10])

net.train(training_set, batch_size=10, epochs=30, lrn_rate=0.5,
          decay_param=5.0, unfriction=0.1, eval_set=test_set,
          save_best=True, print_batches=10)

print(net.test(test_set, print_progress=True)[0])
"""


IMG_FOLDER = "INFO3406_assignment1_query"
NETS = {'-10':  '3072-500-100-10_50',
        '-20':  '3072-500-100-20_33',
        '-100': '3072-500-100-100_24'}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Invocation:")
        print("    python main.py --flag")
        print()
        print("Available flags: ")
        print("    -10  : classify images with CIFAR-10 labels;")
        print("    -20  : classify images into CIFAR-100 coarse labels;")
        print("    -100 : classify images into CIFAR-100 fine labels;")
        print("    -all : classify images with all available categorisations;")
        print()
        print("The result will be output to a csv file with the same name as the flag.")
    elif sys.argv[1] in NETS:
        results = util.eval_images(NETS[sys.argv[1]], IMG_FOLDER)
        util.write_results(results, sys.argv[1][1:] + ".csv")
    elif sys.argv[1] == "-all":
        for flag in NETS:
            results = util.eval_images(NETS[flag], IMG_FOLDER)
            util.write_results(results, flag[1:] + ".csv")
    else:
       print("Unknown flag '{}'.".format(sys.argv[1]))

