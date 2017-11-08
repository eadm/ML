import numpy as np


def gradient_method(train_p, train_c, alpha, eps, kernel):
    # __l0 = np.random.rand(train_c.size)
    __l0 = np.full(train_c.size, 2.5)
    __l = np.copy(__l0)

    k = 0
    for ii in range(0, 100000):
        k = ii
        for j in range(__l0.size):
            __sum = 0.

            for k in range(__l0.size):
                __sum += __l0[k] * kernel(train_p[j], train_p[k]) * train_c[j] * train_c[k]

            __l[j] = min(max(__l0[j] + alpha * (__sum - 1), 0.), 5.)

        if np.abs(__l0 - __l).max() < eps:
            break
        __l0 = np.copy(__l)
        print __l0

    # print i
    return __l
