# import mnist
from nn import NN
import numpy as np


nn = NN()
# for i in range(28):
#     nn.add_empty_layer(768, 768)

# NN.from_file('dmp.txt')

nn.add_empty_layer(2, 4)
nn.add_empty_layer(4, 2)

# print nn.layers
print nn.train(np.array([0.5, 0.5]), np.array([1, 0]))
print nn.train(np.array([-0.5, -0.5]), np.array([1, 0]))

# print nn.layers

for i in range(100000):
    rnd = np.random.rand(2)
    nn.train(rnd, np.array([1, 0]))
    rnd[0] *= -1.
    nn.train(rnd, np.array([0, 1]))
    rnd[1] *= -1.
    nn.train(rnd, np.array([1, 0]))
    rnd[0] *= -1.
    q = nn.train(rnd, np.array([0, 1]))
    if i % 1000 == 0:
        print i, q

nn.to_file('dmp.txt')

print nn.predict(np.array([0.5, 0.5]))
print nn.predict(np.array([-0.5, -0.5]))

# nn.to_file('')
