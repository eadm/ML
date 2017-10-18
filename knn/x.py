from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pylab as pl
import numpy as np
import knn
import ml
import metrics
import kernels
import reader
import transform

points, classes = reader.read_data("chips.txt")


K = 9
KER = 1
POW = 1
metric = (lambda __x1, __x2: metrics.minkowski(__x1, __x2, POW))
kernel = kernels.kernels[KER]

pts, cls = points, classes
# pts, cls = ml.remove_noise(points, classes, metric, kernel, K)

# points = transform.square_poly(points)
# print points
#
# xs = points[:, 0]
# ys = points[:, 1]
# zs = points[:, 2]
#
# fig = pl.figure()
# ax = fig.add_subplot(111, projection='3d')

color_map = ListedColormap(["#012D41", "#1BA5B8", "#FF404E", "#F3B562"])
color_map2 = ListedColormap(["#FF404E", "#1BA5B8"])

# ax.scatter(xs, ys, zs, c=classes, cmap=color_map)


def create_mesh(points, classes, metric, kernel, k):
    xs, ys = points[:, 0], points[:, 1]
    delta = 0.025
    m_x, m_y = np.mgrid[slice(min(xs), max(xs), delta), slice(min(ys), max(ys), delta)]

    container = {}
    container["test_p"] = zip(m_x.ravel(), m_y.ravel())
    container["train_p"] = points
    container["train_c"] = classes

    m_c = knn.validation(container, metric, kernel, k)["knn_c"]
    m_c = np.asarray(m_c).reshape(m_x.shape)
    return m_x, m_y, m_c


m_k, m_p, m_c = create_mesh(pts, cls, metric, kernel, K)
pl.pcolormesh(m_k, m_p, m_c, cmap=color_map)

kxs, kys = points[:, 0], points[:, 1]
pl.scatter(kxs, kys, c=classes, cmap=color_map2)

pl.show()
