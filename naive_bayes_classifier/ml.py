import numpy as np
from model.message import MessageType


def create_cv_from_blocks(blocks):
    folds = []
    for i in range(len(blocks)):
        train = np.concatenate(blocks[:i] + blocks[i + 1:])
        test = blocks[i]
        folds.append({"train": train, "test": test})

    return folds


def create_dict(train_set, count_twice=True):
    __dict = {}
    for message in train_set:
        __used = {}
        for word in np.append(message.subject, message.text):
            __dict.setdefault(word, {"total": 0., "spam": 0., "legit": 0.})

            if not count_twice and word in __used:
                continue
            else:
                __used[word] = True

            __dict[word]["total"] += 1
            if message.type == MessageType.SPAM:
                __dict[word]["spam"] += 1
            else:
                __dict[word]["legit"] += 1

    return __dict


def contingency(cl_orig, cl_test):
    table = {
        "TP": 0.,
        "FP": 0.,
        "FN": 0.,
        "TN": 0.
    }

    for i in range(len(cl_orig)):
        if cl_orig[i] == 0:
            if cl_test[i] == 0:
                table["TN"] += 1
            else:
                table["FP"] += 1
        else:
            if cl_test[i] == 0:
                table["FN"] += 1
            else:
                table["TP"] += 1

    table["P"] = table["TP"] + table["FN"]
    table["N"] = table["TN"] + table["FP"]

    if table["TP"] + table["FP"] == 0.:
        table["PPV"] = 0.
    else:
        table["PPV"] = table["TP"] / (table["TP"] + table["FP"])

    table["ACC"] = (table["TP"] + table["TN"]) / (table["P"] + table["N"])

    if table["P"] == 0.:
        table["TRP"] = 0.
    else:
        table["TRP"] = table["TP"] / table["P"]

    if table["PPV"] + table["TRP"] == 0.:
        table["F1"] = 0.
    else:
        table["F1"] = 2 * table["PPV"] * table["TRP"] / (table["PPV"] + table["TRP"])

    return table
