import math
import numpy as np


def gaussian(x1, x2, sigma):
    return math.exp(-(np.sum(np.square(x1 - x2)) ** 2) / (2 * sigma ** 2))


def scalar(x1, x2):
    return np.sum(x1 * x2)
