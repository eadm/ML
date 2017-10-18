import numpy as np
import math


def square_poly(data):
    return np.array([np.append(x, np.sum(x ** 2)) for x in data])


def polar(data):
    return np.array([[math.sqrt(np.sum(x ** 2)), math.atan(x[1] / x[0])] for x in data])


# def normalise(data):

