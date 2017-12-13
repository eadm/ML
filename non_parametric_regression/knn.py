# import operator
# import ml
# import numpy as np


def classify(train_p, train_c, point, metric, kernel, k):
    data = zip(train_p, train_c)
    nbs = [[metric(point, x[0]), x[0], x[1]] for x in data]
    nbs.sort(key=lambda __x: __x[0])
    stats = 0.
    mult = 0.
    for i in range(k):
        p, c = nbs[i][1], nbs[i][2]
        if nbs[i][0] == nbs[k][0] == 0.:
            window = 0.
        else:
            window = nbs[i][0] / nbs[k][0]
        w = kernel(window)
        mult += w
        stats += c * w

    # print stats
    # print k
    # print "---"
    return stats / mult


def validation(container, metric, kernel, k):
    ct = dict(container)
    knn_c = []
    for p in ct["test_p"]:
        knn_c.append(classify(ct["train_p"], ct["train_c"], p, metric, kernel, k))

    ct["knn_c"] = knn_c
    return ct


# def cv(folds, metric, kernel, k, noise_reduction=False):
#     print "K: %d" % k
#     tests_c = []
#     knn_c = []
#     for fold in folds:
#         if noise_reduction:
#             fold["train_p"], fold["train_c"] = ml.remove_noise(fold["train_p"], fold["train_c"], metric, kernel, k)
#
#         ct = validation(fold, metric, kernel, k)
#         tests_c.extend(ct["test_c"])
#         knn_c.extend(ct["knn_c"])
#
#     table = ml.contingency(tests_c, knn_c)
#     return np.average(table["F1"])
