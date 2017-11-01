import numpy as np


def __get_word_spam_probability(__dict, word, count_s, count_l):

    p_ws = __dict[word]["spam"] / float(count_s)
    p_wl = __dict[word]["legit"] / float(count_l)

    if p_ws != 0:
        return p_ws / float(p_ws + p_wl)

    return -1


def __get_denominators(__dict, words):
    count_s = 0
    count_l = 0

    for word in words:
        count_s += __dict[word]["spam"]
        count_l += __dict[word]["legit"]

    return count_s, count_l


def get_message_spam_probability(__dict, message):
    words = np.concatenate((message.subject, message.text), axis=0)
    count_s, count_l = __get_denominators(__dict, words)

    print count_s, count_l
    ps = []
    bounds = {"lower": [0.1, 0.35], "upper": [0.65, 0.9]}

    for word in words:
        p_sw = __get_word_spam_probability(__dict, word, count_s, count_l)
        if ((p_sw > bounds["lower"][0]) & (p_sw < bounds["lower"][1])) | ((p_sw > bounds["upper"][0]) & (p_sw < bounds["upper"][1])):
            ps.append(p_sw)

    print ps

    mul_p = 1.
    mul_p_1 = 1.
    for p in ps:
        mul_p *= p
        mul_p_1 *= (1 - p)

    return mul_p / float(mul_p + mul_p_1)
