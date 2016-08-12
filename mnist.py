import os, gzip, struct, array, numpy
from scipy.stats import bernoulli

# read: (string, string) -> tuple
def load(image_file, label_file):
    
    with gzip.GzipFile(image_file, 'rb') as infile:
        mag, size, rows, cols = struct.unpack(">IIII", infile.read(16))
        tmp = numpy.array(array.array("B", infile.read()))
        images = numpy.reshape(tmp, (rows*cols, size), order = "F").T
    
    with gzip.GzipFile(label_file, 'rb') as infile:
        mag, size = struct.unpack(">II", infile.read(8))
        labels = numpy.array(array.array("B", infile.read()))

    return (labels, images, (rows, cols))

# squareform: (vec, dim) -> array
def squareform(image_vector, dim):
    return numpy.reshape(image_vector, dim, order = "C")
    
# minibatch: (array, int) -> array
def minibatch(images, size):
    return images[numpy.random.choice(numpy.arange(len(images)), size, replace = False)]

# labeled_minibatch: (array, vec, int, int) -> array
def labeled_minibatch(images, labels, lab, size) -> array:
    numpy.where(labels == lab)
    
# simple_binarize: (array, float) -> array
def simple_binarize(anarray, thresh):
    return numpy.where(anarray>thresh, 1, 0)
    
# stochastic_binarize: (array, float) -> array
def stochastic_binarize(anarray, scale):
    return bernoulli.rvs(anarray/scale)
    
# random_permute: (array) -> array
def random_permute(images):
    return images[numpy.random.permutation(numpy.arange(len(images)))]

if __name__ == "__main__":

    # test load of images
    labels, images, dim = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")

