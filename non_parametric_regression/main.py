import reader
import pylab as pl
import math
import knn
import kernel_smoothing
import numpy as np
import ml


def gaussian_kernel(u):
    return 1. / math.sqrt(2 * math.pi) * math.exp(-0.5 * u**2)


def quartic_kernel(u):
    return 15. / 16. * (1 - u ** 2) ** 2


def quartic_kernel2(u):
    return 3. / 4. * (1 - min(u ** 2, 1.0))


def minkowski(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** 1. / p


kernels = [gaussian_kernel, quartic_kernel, quartic_kernel2]

x, y = reader.read_data('non-parametric.csv')
metric = (lambda __x1, __x2: minkowski(__x1, __x2, 1))

min_mse = 99999999
min_a = []
for i in range(len(kernels)):
    kernel = kernels[i]
    for k in np.arange(0.05, 4., 0.05):  # np.arange(4, 20):
        xs = np.array(x)  # np.arange(min(x), max(x), 0.01)
        ys = []
        for pt in xs:
            ys.append(kernel_smoothing.smooth(x, y, pt, metric, kernel, k))

        mse = ml.mse(y, ys)
        if mse < min_mse:
            min_mse = mse
            min_a = [i, k]
        print mse, i, k

print min_mse
print min_a
xs = np.array(x)  # np.arange(min(x), max(x), 0.01)
ys = []
for pt in xs:
    ys.append(kernel_smoothing.smooth(x, y, pt, metric, kernels[0], 0.8))
    # ys.append(knn.classify(x, y, pt, metric, kernels[0], 7))

pl.plot(xs, ys)
pl.scatter(x, y)
pl.show()
