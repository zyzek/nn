import numpy as np
import scipy.misc as spm
import pickle

# Here for backwards compatibility with some versions of python older than 3.5
OLD = True


def unpickle(file):
    """Unpickle and return an object pickled into a file if it exists."""
    with open(file, 'rb') as fo:
        if not OLD:
            return pickle.load(fo, encoding="bytes")
        else:
            u = pickle._Unpickler(fo)
            u.encoding = 'latin1'
            return u.load()


def save_object(obj, filename='net'):
    """Save an object out to the data/ folder."""
    with open("data/" + filename + ".pkl", 'wb') as fo:
        pickle.dump(obj, fo, pickle.HIGHEST_PROTOCOL)


def load_object(filename='net'):
    """Load an object from the data/ folder."""
    return unpickle("data/" + filename + ".pkl")


def normalised(item):
    n, x = np.min(item), np.max(item)
    return (item - n) / (x - n)

def bmp_to_float(array):
    return array.astype(np.float32).reshape(array.shape + (1,)) / 255.0

def preprocess_cifar(data, labels, num_labels, normalise=True):
    """Given the raw data and labels from a pickled CIFAR data set,
    convert it inot a suitable form for use in a neural network.

    Args:
        data: images as a 2d array of pixel data, ((n, 3072), for example).
        labels: a sequence of n integers in the range [0, num_labels).
        num_labels: the total number of possible categories in the data.
        normalise: if True, stretch an image's pixels to fill up the range [0, 1]
    """

    # Turn the data into floats between 0 and 1.
    #new_data = data.astype(np.float32).reshape(data.shape + (1,)) / 255.0
    new_data = bmp_to_float(data)

    # Generate from the labels neural output sequences.
    new_labels = np.zeros((len(labels), num_labels, 1), np.float32)
    for i, label in enumerate(labels):
        new_labels[i][label] = 1.0

    if normalise:
        for i in range(len(new_data)):
            new_data[i] = normalised(new_data[i])

    return list(zip(new_data, new_labels))


def load_10_batch(num, normalise=True):
    """Load CIFAR-10 batch number num."""
    filename = "data/cifar10/"
    if 1 <= num <= 5:
        filename += "data_batch_{}".format(num)
    else:
        filename += "test_batch"

    result = unpickle(filename)
    if not OLD:
        return preprocess_cifar(result[b'data'], result[b'labels'], 10, normalise)
    else:
        return preprocess_cifar(result['data'], result['labels'], 10, normalise)


def load_100_set(filename, normalise=True):
    """Load the specified CIFAR-100 set with fine-grained labels."""
    result = unpickle("data/cifar100/" + filename)
    if not OLD:
        return preprocess_cifar(result[b'data'], result[b'fine_labels'], 100, normalise)
    else:
        return preprocess_cifar(result['data'], result['fine_labels'], 100, normalise)


def load_20_set(filename, normalise=True):
    """Load the specified CIFAR-100 set with coarse-grained labels."""
    result = unpickle("data/cifar100/" + filename)
    if not OLD:
        return preprocess_cifar(result[b'data'], result[b'coarse_labels'], 20, normalise)
    else:
        return preprocess_cifar(result['data'], result['coarse_labels'], 20, normalise)


def load_test_sequence(folder, normalise=True):
    """Load a sequence of test images from the given folder."""
    images = []

    i = 0

    while True:
        try:
            img = spm.imread(folder + "/img"+["0"+str(i), str(i)][i>9])
            if normalise:
                images.append(normalised(bmp_to_float(img)))
            else:
                images.append(bmp_to_float(img))
            i += 1
        except:
            break

    return images

def eval_images(net_name, image_folder_name):
    net = load_object(net_name)
    test_set = load_test_sequence(image_folder_name)

    results = []

    for example in test_set:
        converted = np.array([example[:,:,c] for c in [0,1,2]]).reshape((3072,1))
        results.append(np.argmax(net.evaluate(converted)))

    return results


def write_results(results, filename):
    import csv
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(results)


def render_image(image, in_shape=(3, 32, 32), out_dim=(512, 512)):
    """Display the given image."""
    spm.toimage(image.reshape(in_shape)).resize(out_dim).show()


def save_image(name, image, in_shape=(3, 32, 32), out_dim=(512, 512)):
    """Save the given image to disk."""
    img = spm.toimage(image.reshape(in_shape)).resize(out_dim)
    spm.imsave(name, img)
