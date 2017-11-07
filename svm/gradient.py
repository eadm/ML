import numpy as np


def gradient_method(train_p, train_c, alpha, eps):
    __w0 = np.random.rand(train_p[0].size)
    __w = np.copy(__w0)

    i = 0
    for ii in range(0, 100000):
        i = ii
        for j in range(0, __w0.size):
            __sum = np.full(train_p[0].size, 0.)

            for i in range(train_c.size - 1):
                __scalar = np.dot(train_p[i], __w0)
                __sum += (train_c[i] - __scalar) * train_p[i]
            __w[j] = __w0[j] + alpha * __sum[j]

        if np.abs(__w0 - __w).max() < eps:
            break
        __w0 = np.copy(__w)

    # print i
    return __w
