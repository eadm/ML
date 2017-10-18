import pylab as pl
import numpy as np
import math
import metrics
import knn
import ml
import reader
import kernels
import transform
from matplotlib.colors import ListedColormap


# def partition(data, percent):  # should be improved
#     left, right = [], []
#     for x in data:
#         if random.random() < percent:
#             left.append(x)
#         else:
#             right.append(x)
#     return left, right

# data = reader.read_data("chips.txt")

# POLAR
# data = map(lambda x: [dist([0, 0], x, 2), math.atan(float(x[1]) / x[0]), x[2]], data)
#
# kxs, kys, vv = map(list, zip(*data))
# x_norm = np.linalg.norm(kxs)
# y_norm = np.linalg.norm(kys)

# data = map(lambda x: [x[0] / x_norm, x[1] / y_norm, x[2]], data)
# END POLAR

color_map = ListedColormap(["#012D41", "#1BA5B8", "#FF404E", "#F3B562"])
color_map2 = ListedColormap(["#FF404E", "#1BA5B8"])

# POWER = 4
# K = 5
# CLASSES = 2
# FOLDS = 10


# def create_mesh(data):
#     xs, ys, cs = map(list, zip(*data))
#     delta = 0.025
#     m_x, m_y = np.mgrid[slice(min(xs), max(xs), delta), slice(min(ys), max(ys), delta)]
    # m_c = map(lambda x: x[2], validation(data, zip(m_x.ravel(), m_y.ravel()), K, POWER, CLASSES))
    # m_c = np.asarray(m_c[len(data):]).reshape(m_x.shape)
    # return m_x, m_y, m_c


points, classes = reader.read_data("chips.txt")
points = transform.square_poly(points)

results = {}
# results [K][Kernel][Power]

FOLDS = []  # range(5, 11, 1)
FOLDS.append(len(points))


K_MAX = int(math.sqrt(len(points))) + 10

ks = [9] # range(2, K_MAX, 1)]
pws = [2]  # range(1, 10, 1)

for k in ks:
    results.setdefault(k, {})

    print "-----------------------------"
    p = 0

    for kernel in kernels.kernels:
        print "---------------"
        print "Kernel: %d" % p
        results[k].setdefault(p, {})

        for power in pws:
            results[k][p].setdefault(power, [])

            print "POW: %d" % power
            metric = (lambda __x1, __x2: metrics.minkowski(__x1, __x2, power))

            for fold_num in FOLDS:
                print "Folds num: %d" % fold_num
                folds = ml.folds(points, classes, fold_num, shuffle=True)

                f1 = knn.cv(folds, metric, kernel, k, noise_reduction=False)
                results[k][p][power].append(f1)
                print f1
        p += 1

results_x = []
for k in results:
    for kernel in results[k]:
        for power in results[k][kernel]:
            results_x.append([np.average(results[k][kernel][power]), k, kernel, power])


print results

results_x.sort(key=lambda __x: __x[0])

print results_x
# tests = [[loo(data, i, POWER, FOLDS), i] for i in range(1, len(data) - (len(data) / FOLDS) - 1)]
#
# best = max(tests)
# print("Best k: %d" % best[1])
# kys, kxs = map(list, zip(*tests))
# pl.plot(kxs, kys)

# (xs, ys, cs, tr_s, te_s, err), k = best

# print("ACC")
# print(loo(data, K, POWER, FOLDS))

# m_k, m_p, m_c = create_mesh(data)
# pl.pcolormesh(m_k, m_p, m_c, cmap=color_map)


# kxs, kys, kc = map(list, zip(*data))
# pl.scatter(kxs, kys, c=kc, cmap=color_map2)

# pl.xlabel("K")
# pl.ylabel("E")
# pl.colorbar()
#
# pl.show()

# print(loo(data, 25, 10))

# pl.scatter(xs, ys, c=cs, cmap=color_map)
# pl.show()
