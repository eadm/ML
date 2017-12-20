import numpy as np


class NN:

    def __init__(self):
        self.layers = []

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
