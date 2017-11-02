import numpy as np
import math
import scipy.special as sp

__EPS = 10 ** -10
__COUNT = 200

__N = 1


def __get_word_spam_probability(__dict, word, h, count_s, count_l):
    if count_s == 0 or count_l == 0:
        return -1.

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


def get_subject_spam_probability(__s_dict, s_h, subject):
    count_s, count_l = __get_denominators(__s_dict, subject)
    ps = []

    for word in subject:
        if word in __s_dict:
            p_sw = __get_word_spam_probability(__s_dict, word, s_h, count_s, count_l)
            ps.append(p_sw)

    ps = bounds(ps)

    return log_union(ps, s_h)


def get_body_spam_probability(__b_dict, b_h, body):
    count_s, count_l = __get_denominators(__b_dict, body)
    ps = []

    for word in body:
        if word in __b_dict:
            p_sw = __get_word_spam_probability(__b_dict, word, b_h, count_s, count_l)
            ps.append(p_sw)

    ps = bounds(ps)

    return log_union(ps, b_h)


def get_message_spam_probability(__s_dict, __b_dict, s_h, b_h, h, message):
    s_p = get_subject_spam_probability(__s_dict, s_h, message.subject)
    b_p = get_body_spam_probability(__b_dict, b_h, message.text)

    p = s_p * b_p / (s_p * b_p + (1 - s_p) * (1 - b_p) * h ** 2)
    return p


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
    h = h ** (1 - len(ps))
    for p in ps:
        h *= (1 - p) / p

    return 1. / (1. + h)


def log_union(ps, h):
    ln_p = math.log(1 / h)
    for p in ps:
        ln_p += math.log(p / (1 - p))

    if ln_p < 0:
        return 0.05
    else:
        return 0.95

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
