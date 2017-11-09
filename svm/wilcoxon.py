import numpy as np


def get_p_value(x, y):

    xy = x - y
    n = 0

    for i in range(len(xy)):
        if np.sign(xy[i]) != 0:
            n += 1

    r = (n + 1) / 2.
    r_plus = r_minus = 0

    for i in range(len(xy)):
        if np.sign(xy[i]) > 0:
            r_plus += r
        if np.sign(xy[i]) < 0:
            r_minus += r

    t = min(r_plus, r_minus)
    mn = n * (n + 1) / 4.
    se = n * (n + 1) * (2. * n + 1)

    se = np.sqrt(se / 24)
    z = (t - mn) / se
    p_value = 2. * abs(z)

    return p_value
