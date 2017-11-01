import numpy as np


def __get_word_spam_probability(__dict, word, count_s, count_l):
    p_ws = __dict[word]["spam"] / float(count_s)
    p_wl = __dict[word]["legit"] / float(count_l)

    if p_ws + p_wl != 0:
        return p_ws / float(p_ws + p_wl)

    return -1


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

    print count_s, count_l
    ps = []
    bounds = {"lower": [0.1, 0.35], "upper": [0.65, 0.9]}

    for word in words:
        if word in __dict:
            p_sw = __get_word_spam_probability(__dict, word, count_s, count_l)
            if ((p_sw > bounds["lower"][0]) & (p_sw < bounds["lower"][1])) | (
                        (p_sw > bounds["upper"][0]) & (p_sw < bounds["upper"][1])):
                ps.append(p_sw)

    print ps

    c = 1.
    for p in ps:
        c *= (1 - p) / p

    return 1. / (1. + c)
