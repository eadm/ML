import math


def one(_):
    return 1.


def none(u):
    return 1 - abs(u)


def parabolic(u):
    return 3. / 4. * (1 - u ** 2)


def tricube(u):
    return 70.0 / 80.0 * (1 - abs(u) ** 3) ** 3


def cosine(u):
    return math.pi / 4 * math.cos(math.pi / 2 * u)


kernels = [one, none, parabolic, tricube, cosine]
