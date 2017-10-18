from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import reader

xs, ys = reader.read_data("prices.txt")
x1s = xs[:, 0]
x2s = xs[:, 1]

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1s, x2s, ys)

# tgt function is y = w0 + w * x + e



pl.show()
