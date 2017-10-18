import numpy as np


def mse(__a, __b):
    return np.square(__a - __b).mean()


def mse_w(__w, __x, __y):
    return np.square(np.dot(__w, __x) - __y).mean()


def mse_w_derivative(__w, __x, __y):
    return 2


def __cut_from_array(array, start, end):
    return np.concatenate((array[:start], array[end:])), array[start:end]


def folds(points, classes, folds_num, shuffle=False):
    fds = []

    if shuffle:
        shape = points.shape
        data = np.zeros((shape[0], shape[1] + 1))
        data[:, :-1] = points
        data[:, -1] = classes

        np.random.shuffle(data)
        points = data[:, :-1]
        classes = data[:, -1]

    size = len(points) / folds_num
    for start in range(0, len(points), size):
        train_p, test_p = __cut_from_array(points, start, start + size)
        train_c, test_c = __cut_from_array(classes, start, start + size)
        fds.append({"train_p": train_p, "train_c": train_c, "test_p": test_p, "test_c": test_c})

    return fds
