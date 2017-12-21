import numpy as np


def mse(__a, __b):
    return np.square(__a - __b).mean()


def rmse(__a, __b):
    return np.sqrt(mse(__a, __b))
