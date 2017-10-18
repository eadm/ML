import numpy as np


def mse(__a, __b):
    return np.square(__a - __b).mean()


def mse_w(__w, __x, __y):
    return np.square(np.dot(__w, __x) - __y).mean()


def mse_w_derivative(__w, __x, __y):
    return 2
