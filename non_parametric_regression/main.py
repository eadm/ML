import reader
import pylab as pl
import math
import knn
import numpy as np
import ml


def gaussian_kernel(u):
    return 1. / math.sqrt(2 * math.pi) * math.exp(-0.5 * u**2) / 0.35


def quartic_kernel(u):
    return (1 - u ** 2) ** 2


def minkowski(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** 1. / p


kernels = [gaussian_kernel, quartic_kernel]

x, y = reader.read_data('non-parametric.csv')
metric = (lambda __x1, __x2: minkowski(__x1, __x2, 1))

min_mse = 99999999
min_a = []
# for i in range(len(kernels)):
#     kernel = kernels[i]
#     for k in range(1, 50):
#         xs = np.array(x)  # np.arange(min(x), max(x), 0.01)
#         ys = []
#         for pt in xs:
#             ys.append(knn.classify(x, y, pt, metric, kernel, k))
#
#         mse = ml.mse(y, ys)
#         if mse < min_mse:
#             min_mse = mse
#             min_a = [i, k]
#         print mse

print min_mse
print min_a
xs = np.array(x)  # np.arange(min(x), max(x), 0.01)
ys = []
for pt in xs:
    ys.append(knn.classify(x, y, pt, metric, gaussian_kernel, 10))

pl.plot(xs, ys)
pl.scatter(x, y)
pl.show()
