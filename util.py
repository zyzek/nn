import numpy as np
import scipy.misc as spm
import pickle
import gzip

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding="bytes")


def save_object(obj, filename='wts'):
    with open("data/weights/" + filename + ".pkl", 'wb') as fo:
        pickle.dump(obj, fo, pickle.HIGHEST_PROTOCOL)


def preprocess_cifar_10(data, labels):
    new_data = data.astype(np.float32).reshape(data.shape + (1,)) / 255.0
    new_labels = np.zeros((len(labels), 10, 1), np.float32)
    for i, label in enumerate(labels):
        new_labels[i][label] = 1.0

    return list(zip(new_data, new_labels))


def load_10_batch(num):
    filename = "data/cifar10/"
    if 1 <= num <= 5:
        filename += "data_batch_{}".format(num)
    else:
        filename += "test_batch"

    result = unpickle(filename)
    return preprocess_cifar_10(result[b'data'], result[b'labels'])


def load_mnist():
    with gzip.open("data/mnist.pkl.gz", 'rb') as fo:
        training, validation, test = pickle.load(fo, encoding="bytes")

        training_labels = np.zeros((len(training[1]), 10, 1), np.float32)
        for i, label in enumerate(training[1]):
            training_labels[i][label] = 1.0

        test_labels = np.zeros((len(test[1]), 10, 1), np.float32)
        for i, label in enumerate(test[1]):
            test_labels[i][label] = 1.0

        return (list(zip(training[0], training_labels)), list(zip(test[0], test_labels)))


def render_image(image):
    spm.toimage(image.reshape(3, 32, 32)).resize((512,512)).show()
