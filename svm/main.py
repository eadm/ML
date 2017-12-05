import reader
import ml
import kernels
import numpy as np
import opt

FOLDS = 10
points, classes = reader.read_data("chips.txt")
folds = ml.folds(points, classes, FOLDS, shuffle=True)
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
    slv = opt.solve_sp(fold["train_p"], fold["train_c"], lambda x, y: kernels.gaussian(x, y, sigma), c).x

    print slv
    w = get_w(slv, fold["train_p"], fold["train_c"])
    b = get_b(w, fold["train_p"], fold["train_c"])

    print w, b

    for p in fold["test_p"]:
        ans_c.append(classify(p, slv, fold["train_p"], fold["train_c"], c, b, EPS))
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
        ans_p.extend(fold["train_c"])

    ctg = ml.contingency(ans_p, ans_c)
    print ctg
    return ctg


def classify(p, ls, train_p, train_c, c, b, eps):
    __sum = 0.0
    for i in range(ls.size):
        if eps < ls[i] < c - eps:
            print "----"
            print ls[i]
            print train_p[i]
            print train_c[i]
            __sum += np.dot(train_p[i], p) * ls[i] * train_c[i]
    return int(np.sign(__sum))


# check_fold(folds[0], 2.5, 1.2)
