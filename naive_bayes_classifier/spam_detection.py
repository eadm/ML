import numpy as np
import math
import scipy.special as sp

__EPS = 10 ** -6
__COUNT = 10

__N = 1


def __get_word_spam_probability(__dict, word, h, count_s, count_l):
    p_ws = __dict[word]["spam"] / float(count_s)
    p_wl = __dict[word]["legit"] / float(count_l)

    d = p_ws + (p_wl * h)
    if d != 0:
        return p_ws / float(d)

    return -1.


def __get_denominators(__dict, words):
    count_s = 0
    count_l = 0

    for word in words:
        if word in __dict:
            count_s += __dict[word]["spam"]
            count_l += __dict[word]["legit"]

    return count_s, count_l


def get_message_spam_probability(__dict, h, message):
    words = np.concatenate((message.subject, message.text), axis=0)
    count_s, count_l = __get_denominators(__dict, words)

    # print count_s, count_l
    ps = []

    for word in words:
        if word in __dict:
            p_sw = __get_word_spam_probability(__dict, word, h, count_s, count_l)
            ps.append(p_sw)

    ps = bounds(ps)

    return naive_union(ps, h ** (1 - len(ps)))


def none(ps):
    return [val for val in ps if abs(abs(val) - 1.) > __EPS and abs(val) > __EPS]


def bounds(ps):
    __bounds = {"lower": [0.001, 0.154], "upper": [0.856, 0.999]}
    ps = [p_sw for p_sw in ps if ((p_sw > __bounds["lower"][0]) and (p_sw < __bounds["lower"][1])) or (
            (p_sw > __bounds["upper"][0]) and (p_sw < __bounds["upper"][1]))]
    return ps


def tops(ps):
    ps = [val for val in ps if abs(abs(val) - 1.) > __EPS and abs(val) > __EPS]
    ps = sorted(ps, key=lambda __x: 0.5 - abs(__x - 0.5))
    ps = ps[:__COUNT]
    return ps


def naive_union(ps, h):
    for p in ps:
        h *= (1 - p) / p

    return 1. / (1. + h)

# def chi_union(ps):
#     ln = 0
#     for p in ps:
#         ln += math.log(p)
#
#     if ln == 0:
#         return 0.5
#     x = -2 * ln
#     v = 2 * __N
#     f = (2 ** (-v / 2)) * (x ** ((-v / 2) - 1)) * (math.e ** (-1 / (2 * x))) / sp.gamma(v / 2)
#     print f
#     return f
