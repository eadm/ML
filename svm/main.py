import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

import ml
import reader
from svm import SVM

FOLDS = 10
points, classes = reader.read_data("chips.txt")

color_map = ListedColormap(["#012D41", "#1BA5B8", "#FF404E", "#F3B562"])
color_map2 = ListedColormap(["#FF404E", "#1BA5B8"])


def show_plot():
    m_k, m_p, m_c = create_mesh(points, classes)
    pl.pcolormesh(m_k, m_p, m_c, cmap=color_map)

    kxs, kys = points[:, 0], points[:, 1]
    pl.scatter(kxs, kys, c=classes, cmap=color_map2)

    pl.show()


def create_mesh(_points, _classes):
    xs, ys = points[:, 0], points[:, 1]
    delta = 0.025
    m_x, m_y = np.mgrid[slice(min(xs), max(xs), delta), slice(min(ys), max(ys), delta)]

    svm = SVM(C=10, gamma=10.0)
    svm.fit(_points, _classes)

    _m_c = svm.predict(np.array(zip(m_x.ravel(), m_y.ravel())))
    _m_c = np.asarray(_m_c).reshape(m_x.shape)
    return m_x, m_y, _m_c


def check():
    folds = ml.folds(points, classes, FOLDS, shuffle=True)
    svm = SVM(C=10, gamma=2.0)

    ans_c = []
    ans_p = []
    for fold in folds:
        svm.fit(fold["train_p"], fold["train_c"])
        c = fold["test_c"]
        p = svm.predict(fold["test_p"])
        ans_c.extend(c)
        ans_p.extend(p)
        print ml.contingency(p, c)

    ctg = ml.contingency(ans_p, ans_c)
    print ctg


# check()
show_plot()
