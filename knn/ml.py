import numpy as np
import knn


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


def __cut_from_array(array, start, end):
    return np.concatenate((array[:start], array[end:])), array[start:end]


def remove_noise(points, classes, metric, kernel, k):
    fds = folds(points, classes, len(points))
    pts, cls = [], []

    for fold in fds:
        cl = knn.classify(fold["train_p"], fold["train_c"], fold["test_p"][0], metric, kernel, k)
        if cl == fold["test_c"][0]:  # not noise
            pts.append(fold["test_p"][0])
            cls.append(cl)

    return np.array(pts), np.array(cls)


def folds(points, classes, folds_num, shuffle=False):
    fds = []

    if shuffle:
        shape = points.shape
        data = np.zeros((shape[0], shape[1] + 1))
        data[:, :-1] = points
        data[:, -1] = classes

        np.random.shuffle(data)
        points = data[:, :-1]
        classes = data[:, -1]

    size = len(points) / folds_num
    for start in range(0, len(points), size):
        train_p, test_p = __cut_from_array(points, start, start + size)
        train_c, test_c = __cut_from_array(classes, start, start + size)
        fds.append({"train_p": train_p, "train_c": train_c, "test_p": test_p, "test_c": test_c})

    return fds
