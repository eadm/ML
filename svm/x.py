import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

import reader
from svm import SVM

points, classes = reader.read_data("chips.txt")

color_map = ListedColormap(["#012D41", "#1BA5B8", "#FF404E", "#F3B562"])
color_map2 = ListedColormap(["#FF404E", "#1BA5B8"])


def create_mesh(_points, _classes):
    xs, ys = points[:, 0], points[:, 1]
    delta = 0.025
    m_x, m_y = np.mgrid[slice(min(xs), max(xs), delta), slice(min(ys), max(ys), delta)]


    # clf = o_svm.NuSVC()
    # clf.fit(_points, _classes)

    # _m_c = clf.predict(zip(m_x.ravel(), m_y.ravel()))

    o_svm = SVM()
    o_svm.fit(_points, _classes)
    _m_c = o_svm.predict(np.array(zip(m_x.ravel(), m_y.ravel())))

    # o_svm = ts.Svm(2.8, 1.2)
    # o_svm.fit(_points, _classes)
    # _m_c = o_svm.predict(zip(m_x.ravel(), m_y.ravel()))

    _m_c = np.asarray(_m_c).reshape(m_x.shape)
    return m_x, m_y, _m_c


m_k, m_p, m_c = create_mesh(points, classes)
pl.pcolormesh(m_k, m_p, m_c, cmap=color_map)

kxs, kys = points[:, 0], points[:, 1]
pl.scatter(kxs, kys, c=classes, cmap=color_map2)

pl.show()
