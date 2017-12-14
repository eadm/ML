import numpy as np

import svm.kernels
import svm.ml
import svm.reader
from svm.obsolete import opt

FOLDS = 10
points, classes = svm.reader.read_data("chips.txt")
folds = svm.ml.folds(points, classes, FOLDS, shuffle=True)
EPS = 0.0001


def get_w(ls, train_p, train_c):
    __w = np.zeros(train_p[0].size)
    for i in range(train_c.size):
        __w += ls[i] * train_p[i] * train_c[i]
    return __w


def get_b(w, train_p, train_c):
    return np.dot(w, train_p[4]) - train_c[4]


def check_fold(fold, c, sigma):
    ans_c = []

    def kernel(x, y):
        return svm.kernels.gaussian(x, y, sigma)
    slv = opt.solve_sp(fold["train_p"], fold["train_c"], kernel, c).x

    print slv
    w = get_w(slv, fold["train_p"], fold["train_c"])
    b = get_b(w, fold["train_p"], fold["train_c"])

    print w, b

    for p in fold["test_p"]:
        ans_c.append(classify(p, slv, fold["train_p"], fold["train_c"], kernel, c, b, EPS))
    return ans_c


def check(_folds, c, sigma):
    print "--------------------"
    print "C:"
    print c
    print "Sigma:"
    print sigma
    ans_c = []
    ans_p = []
    for fold in _folds:
        ans_c.extend(check_fold(fold, c, sigma))
        ans_p.extend(fold["test_c"])

    ctg = svm.ml.contingency(ans_p, ans_c)
    print ctg
    return ctg


def classify(p, ls, train_p, train_c, ker, c, b, eps):
    __sum = 0.0
    for i in range(ls.size):
        if eps < ls[i] < c - eps:
            # print "----"
            # print ls[i]
            # print train_p[i]
            # print train_c[i]
            __sum += ker(train_p[i], p) * ls[i] * train_c[i]
    return int(np.sign(__sum))


check_fold(folds[0], 2.5, 1.2)
# check(folds, 0.8, 0.5)
