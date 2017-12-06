from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pylab as pl
import numpy as np
import reader
from sklearn import svm

points, classes = reader.read_data("chips.txt")


color_map = ListedColormap(["#012D41", "#1BA5B8", "#FF404E", "#F3B562"])
color_map2 = ListedColormap(["#FF404E", "#1BA5B8"])


def create_mesh(_points, _classes):
    xs, ys = points[:, 0], points[:, 1]
    delta = 0.025
    m_x, m_y = np.mgrid[slice(min(xs), max(xs), delta), slice(min(ys), max(ys), delta)]

    clf = svm.NuSVC()
    clf.fit(_points, _classes)

    _m_c = clf.predict(zip(m_x.ravel(), m_y.ravel()))
    _m_c = np.asarray(_m_c).reshape(m_x.shape)
    return m_x, m_y, _m_c


m_k, m_p, m_c = create_mesh(points, classes)
pl.pcolormesh(m_k, m_p, m_c, cmap=color_map)

kxs, kys = points[:, 0], points[:, 1]
pl.scatter(kxs, kys, c=classes, cmap=color_map2)

pl.show()
