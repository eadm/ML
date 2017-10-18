import numpy as np


def minkowski(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** 1. / p
