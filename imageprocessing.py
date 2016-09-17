import numpy as np
import scipy.misc as spm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
        return dict

def load_10_batch(num):
    filename = "data/cifar10/"
    if 1 <= num <= 5:
        filename += "data_batch_{}".format(num)
    else:
        filename += "test_batch"

    return unpickle(filename)

def render_image(image):
    spm.toimage(image.reshape(3, 32, 32)).resize((512,512)).show()

def preprocess_cifar_10(data, labels):
    new_data = data.astype(np.float32).reshape(data.shape + (1,)) / 255.0
    new_labels = np.zeros((len(labels), 10, 1), np.float32)
    for i, label in enumerate(labels):
        new_labels[i][label] = 1.0

    return list(zip(new_data, new_labels))






