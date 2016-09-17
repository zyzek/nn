import network
import numpy as np
import imageprocessing as ip

image_dict = ip.load_10_batch(1)
processed = ip.preprocess_cifar_10(image_dict[b'data'], image_dict[b'labels'])

net = network.BackpropNetwork([32 * 32 * 3, 100, 50, 30, 10])

test_dict = ip.load_10_batch(1)
test_set = ip.preprocess_cifar_10(test_dict[b'data'], test_dict[b'labels'])

net.train(processed, batch_size=1000, epochs=1, lrn_rate=1.0, print_batches=10)

correct = 0
results = [0]*10
for i, example in enumerate(test_set):
    result = net.evaluate(example[0])
    if (i % 1000) == 0:
        print(result)
        print(np.argmax(result))
    predicted = np.argmax(result)
    expected = np.argmax(example[1])


    results[predicted] += 1
    if predicted == expected:
        correct += 1

print(results)
print("{}/{}".format(correct, len(test_set)))
