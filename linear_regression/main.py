from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import pylab as pl
import numpy as np
import gradient as gr
import ml
import reader

xs, ys = reader.read_data("prices.txt")
ys = ys / np.linalg.norm(ys)
xs = normalize(xs, axis=0, norm="l1")

x1s = xs[:, 0]
x2s = xs[:, 1]

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')


# tgt function is y = w0 + w * x + e
fds = ml.folds(xs, ys, 10)

fold = fds[0]
alpha = 0.2
w = gr.gradient_method(fold["train_p"], fold["train_c"], alpha)
print w


X = np.arange(0, 0.05, 0.001)
Y = np.arange(0, 0.03, 0.001)
X, Y = np.meshgrid(X, Y)
Z = X * w[0] + Y * w[1]
ax.plot_surface(X, Y, Z)

ax.scatter(x1s, x2s, ys)

pl.show()
