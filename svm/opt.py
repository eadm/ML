import cvxopt
import numpy as np
from scipy.optimize import minimize


def __compute_kernel(train_p, kernel, n):
    K = np.zeros((n, n))
    for i, x_i in enumerate(train_p):
        for j, x_j in enumerate(train_p):
            K[i, j] = kernel(x_i, x_j)
    return K


def __svm(l, train_p, train_c, kernel):
    s = 0.
    for i in range(l.size):
        for j in range(l.size):
            s += l[i] * l[j] * train_c[i] * train_c[j] * kernel(train_p[i], train_p[j])

    return -l.sum() + 0.5 * s


def solve_sp(train_p, train_c, kernel, C):
    def f(__x):
        return __svm(__x, train_p, train_c, kernel)

    bnds = [(0., C)] * train_c.size

    def con(__x):
        return np.dot(__x, train_c)

    return minimize(f, np.zeros(train_c.size), method='SLSQP', bounds=bnds, constraints={"type": "eq", "fun": con})


def solve(train_p, train_c, kernel, C):
    n = len(train_c)
    P = cvxopt.matrix(np.outer(train_c, train_c) * __compute_kernel(train_p, kernel, n))
    q = cvxopt.matrix(-1 * np.ones(n))

    A = cvxopt.matrix(train_c, (1, n))
    b = cvxopt.matrix(0.0)

    G_std = cvxopt.matrix(np.diag(np.ones(n) * -1))
    h_std = cvxopt.matrix(np.zeros(n))

    G_slack = cvxopt.matrix(np.diag(np.ones(n)))
    h_slack = cvxopt.matrix(np.ones(n) * C)

    G = cvxopt.matrix(np.vstack((G_std, G_slack)))
    h = cvxopt.matrix(np.vstack((h_std, h_slack)))

    print P
    print q
    print G

    print h
    print A
    print b

    return cvxopt.solvers.qp(P, q, G, h, A, b)["x"]
