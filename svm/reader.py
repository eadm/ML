import numpy as np


def __convert_class(c):
    if c == 0:
        return -1.
    else:
        return 1.


def read_data(path):
    with open(path) as f:
        data = f.readlines()

    data = [[float(y) for y in x.strip().split(',')] for x in data]
    data = map(lambda x: [x[0], x[1], __convert_class(int(x[2]))], data)

    kxs, kys, classes = map(list, zip(*data))
    return np.array(zip(kxs, kys)), np.array(classes)
