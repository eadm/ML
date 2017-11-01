import numpy as np


__EPS = 10 ** -10
__COUNT = 5


def __get_word_spam_probability(__dict, word, count_s, count_l):
    p_ws = __dict[word]["spam"] / float(count_s)
    p_wl = __dict[word]["legit"] / float(count_l)

    if p_ws + p_wl != 0:
        return p_ws / float(p_ws + p_wl)

    return -1.


def __get_denominators(__dict, words):
    count_s = 0
    count_l = 0

    for word in words:
        if word in __dict:
            count_s += __dict[word]["spam"]
            count_l += __dict[word]["legit"]

    return count_s, count_l


def get_message_spam_probability(__dict, message):
    words = np.concatenate((message.subject, message.text), axis=0)
    count_s, count_l = __get_denominators(__dict, words)

    # print count_s, count_l
    ps = []

    for word in words:
        if word in __dict:
            p_sw = __get_word_spam_probability(__dict, word, count_s, count_l)
            # if ((p_sw > bounds["lower"][0]) and (p_sw < bounds["lower"][1])) or (
            #             (p_sw > bounds["upper"][0]) and (p_sw < bounds["upper"][1])):
            ps.append(p_sw)

    ps = bounds(ps)

    c = 1.
    for p in ps:
        c *= (1 - p) / p

    return 1. / (1. + c)


def bounds(ps):
    __bounds = {"lower": [0.001, 0.1], "upper": [0.9, 0.999]}
    ps = [p_sw for p_sw in ps if ((p_sw > __bounds["lower"][0]) and (p_sw < __bounds["lower"][1])) or (
                        (p_sw > __bounds["upper"][0]) and (p_sw < __bounds["upper"][1]))]
    return ps


def tops(ps):
    ps = [val for val in ps if abs(abs(val) - 1.) > __EPS and abs(val) > __EPS]
    ps = sorted(ps, key=lambda __x: 0.5 - abs(__x - 0.5))
    ps = ps[__COUNT:]
    return ps
