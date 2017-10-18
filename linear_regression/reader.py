import numpy as np


def read_data(path):
    with open(path) as f:
        data = f.readlines()

    data = [[int(__y) for __y in __x.strip().split(',')] for __x in data]

    kx1, kx2, ys = map(list, zip(*data))
    return np.array(zip(kx1, kx2)), np.array(ys)
