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
C = 1.1
EPS = 0.0000001

slv = opt.solve_sp(folds[0]["train_p"], folds[0]["train_c"], lambda x, y: kernels.gaussian(x, y, 0.5), C)
slv = slv.x
print slv
# l = gradient(folds[0]["train_p"], folds[0]["train_c"], 0.1, 0.0001, lambda x, y: kernels.gaussian(x, y, 1.0))


def classify(p, ls, train_p, train_c):
    __sum = 0.0
    for i in range(ls.size):
        if ls[i] != C and ls[i] > EPS:
            __sum += np.dot(train_p[i], p) * ls[i] * train_c[i]
    return int(np.sign(__sum))


ans_c = []
for p in folds[0]["test_p"]:
    ans_c.append(classify(p, slv, folds[0]["train_p"], folds[0]["train_c"]))

print ans_c
print ml.contingency(folds[0]["test_c"], ans_c)

