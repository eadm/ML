import numpy as np
import kernels
from optimizer import optimize

tolerance = 1e-4
iterate_limit = 10000
passes_limit = 10


class SVM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.support_vectors = X
        self.y = y

        K = kernels.rbf(self.gamma)(X, None)

        self.coefficients = np.zeros(X.shape[0])
        self.intercept = optimize(K, y, self.coefficients, self.C, tolerance, passes_limit, iterate_limit)

        support_vectors = np.nonzero(self.coefficients)
        self.coefficients = self.coefficients[support_vectors]
        self.support_vectors = X[support_vectors]
        self.y = y[support_vectors]

        return self

    def predict(self, X):
        K = kernels.rbf(self.gamma)(X, self.support_vectors)
        cl = (self.intercept + np.sum(self.coefficients[np.newaxis, :] * self.y * K, axis=1))
        return np.sign(cl)
