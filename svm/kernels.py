import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def rbf(gamma):
    kernel_args = {"gamma": gamma}
    return lambda X, Y: pairwise_kernels(X, Y=Y, metric="rbf", **kernel_args)


def gaussian(x1, x2, sigma):
    return math.exp(-(np.sum(np.square(x1 - x2)) ** 2) / (2 * sigma ** 2))


def scalar(x1, x2):
    return np.inner(x1, x2)


def scalar2(x1, x2):
    return np.dot(x1, x2) ** 2
