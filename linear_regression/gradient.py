import numpy as np


def gradient_method(train_p, train_c, alpha):
    __w0 = np.full(train_p[0].size, 1.)
    __w = np.copy(__w0)

    for _ in range(0, 1000):
        for j in range(0, __w0.size):
            __sum = np.full(train_p[0].size, 0.)

            for i in range(train_c.size - 1):
                __scalar = np.dot(train_p[i], __w0)
                __sum += (train_c[i] - __scalar) * train_p[i]
            __w[j] = __w0[j] + alpha * __sum[j]

        __w0 = np.copy(__w)

    return __w
