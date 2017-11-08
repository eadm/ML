import reader
import ml
import kernels
import numpy as np
from gradient import gradient_method as gradient

FOLDS = 10

points, classes = reader.read_data("chips.txt")
# print points
# print classes

folds = ml.folds(points, classes, FOLDS, shuffle=True)

# print folds

l = gradient(folds[0]["train_p"], folds[0]["train_c"], 0.1, 0.0001, lambda x, y: kernels.gaussian(x, y, 1.0))
w = np.zeros(folds[0]["train_p"][0].size)
for i in range(l.size):
    w += l[i] * folds[0]["train_c"][i] * folds[0]["train_p"][i]
# w = np.sum(l * folds[0]["train_c"] * folds[0]["train_p"])
print l
print w

ans_c = []
for p in folds[0]["test_p"]:
    ans_c.append(int(np.sign(np.dot(w, p))))

print ans_c
print ml.contingency(folds[0]["test_c"], ans_c)
#
