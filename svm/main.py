import reader
import ml
import kernels
import numpy as np
import opt

FOLDS = 10

points, classes = reader.read_data("chips.txt")
# print points
# print classes

folds = ml.folds(points, classes, FOLDS, shuffle=True)

# print folds
# C = 2.5
EPS = 0.0001

# slv =
# slv = slv.x
# print slv
# l = gradient(folds[0]["train_p"], folds[0]["train_c"], 0.1, 0.0001, lambda x, y: kernels.gaussian(x, y, 1.0))


def get_w(ls, train_p, train_c):
    __w = np.zeros(train_p[0].size)
    for i in range(train_c.size):
        __w += ls[i] * train_p[i] * train_c[i]
    return __w


def get_b(w, train_p, train_c):
    return np.dot(w, train_p[4]) - train_c[4]


def check_fold(fold, c, sigma):
    ans_c = []
    slv = opt.solve_sp(fold["train_p"], fold["train_c"], lambda x, y: kernels.gaussian(x, y, sigma), c).x

    w = get_w(slv, fold["train_p"], fold["train_c"])
    b = get_b(w, fold["train_p"], fold["train_c"])

    for p in fold["test_p"]:
        ans_c.append(classify(p, slv, fold["train_p"], fold["train_c"], c, b))
    return ans_c


def check(_folds, c, sigma):
    ans_c = []
    ans_p = []
    for fold in _folds:
        ans_c.extend(check_fold(fold, c, sigma))
        ans_p.extend(fold["train_c"])

    return ml.contingency(ans_p, ans_c)


def classify(p, ls, train_p, train_c, c,  b):
    __sum = 0.0
    for i in range(ls.size):
        if EPS < ls[i] < c - EPS:
            print ls[i]
            __sum += np.dot(train_p[i], p) * ls[i] * train_c[i]
    return int(np.sign(__sum))

