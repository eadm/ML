import numpy as np
import kernels
import opt
import main

EPS = 0.001


class Svm:
    def __init__(self, c, sigma):
        self.c = c
        self.sigma = sigma

        self.points = 0
        self.classes = 0
        self.slv = 0
        self.b = 0
        self.w = 0

    def kernel(self, x, y):
        return kernels.gaussian(x, y, self.sigma)

    def fit(self, points, classes):
        self.points = points
        self.classes = classes
        slv = opt.solve_sp(points, classes, self.kernel, self.c).x
        print slv

        w = main.get_w(slv, points, classes)
        b = main.get_b(w, points, classes)

        self.slv = slv
        self.b = b
        self.w = w

    def predict(self, points):
        classes = []
        for p in points:
            classes.append(main.classify(p, self.slv, self.points, self.classes, self.kernel, self.c, self.b, EPS))

        classes = np.array(classes)
        print classes
        return classes
