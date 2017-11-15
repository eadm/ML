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
C = 2.5
EPS = 0.0001

slv = opt.solve_sp(folds[0]["train_p"], folds[0]["train_c"], lambda x, y: kernels.gaussian(x, y, 0.5), C)
slv = slv.x
print slv
# l = gradient(folds[0]["train_p"], folds[0]["train_c"], 0.1, 0.0001, lambda x, y: kernels.gaussian(x, y, 1.0))


def getW(ls, train_p, train_c):
    __w = np.zeros(train_p[0].size)
    for i in range(train_c.size):
        __w += ls[i] * train_p[i] * train_c[i]
    return __w


def getB(w, train_p, train_c):
    return np.dot(w, train_p[4]) - train_c[4]


w = getW(slv, folds[0]["train_p"], folds[0]["train_c"])
b = getB(w, folds[0]["train_p"], folds[0]["train_c"])
print w
print b


def classify(p, ls, train_p, train_c, b):
    __sum = 0.0
    for i in range(ls.size):
        if EPS < ls[i] < C - EPS:
            print ls[i]
            __sum += np.dot(train_p[i], p) * ls[i] * train_c[i]
    return int(np.sign(__sum))


ans_c = []
for p in folds[0]["test_p"]:
    ans_c.append(classify(p, slv, folds[0]["train_p"], folds[0]["train_c"], b))

print ans_c
print ml.contingency(folds[0]["test_c"], ans_c)

