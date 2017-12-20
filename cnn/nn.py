import numpy as np
from math import exp


class NN:
    def __init__(self):
        self.layers = []
        self.sigma = np.vectorize(lambda x: 1. / (1. + exp(-x)))
        self.sigma1 = lambda x: self.sigma(x) * (1. - self.sigma(x))
        self.nu = 3.

    def add_layer(self, matrix):
        self.layers.append(matrix)

    def add_empty_layer(self, x, y):
        self.layers.append(np.random.rand(x, y) * (1. / x) - (1. / 2. / x))

    def to_file(self, path):
        f = open(path, 'w')

        for i in range(len(self.layers)):
            layer = self.layers[i]
            for l in layer:
                f.write(' '.join(map(str, l)))
                f.write('\n')
            f.write('|')
            if i + 1 != len(self.layers):
                f.write('\n')

    def predict(self, x):
        for layer in self.layers:
            x = self.sigma(np.dot(x, layer))
        return x

    def train(self, x, y):
        y1 = x
        ys1 = [x]
        for layer in self.layers:
            y1 = self.sigma(np.dot(y1, layer))
            ys1.append(y1)

        es = [y1 - self.sigma(y)]
        q = np.sum(es[-1] ** 2)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            s1 = self.sigma1(ys1[i + 1])
            eps = np.dot(layer, es[-1] * s1)
            es.append(eps)
        es = list(reversed(es))

        for k in range(len(self.layers)):
            layer = self.layers[k]

            y1 = ys1[k]
            s1 = self.sigma1(y1)
            for i in range(len(layer)):
                for j in range(len(layer[i])):
                    layer[i][j] = layer[i][j] - self.nu * (es[k][i] * s1[i] * ys1[k + 1][j])

        return q

    @staticmethod
    def from_file(path):
        with open(path) as f:
            lines = f.readlines()

        nn = NN()
        tmp = []
        for i in range(len(lines)):
            if '|' in lines[i]:
                nn.add_layer(np.array(tmp))
                tmp = []
            else:
                tmp.append(map(float, lines[i].split(' ')))

        return nn
