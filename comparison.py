import wilcoxon
import numpy as np
from knn import metrics, kernels
from svm import reader, ml
from svm.svm import SVM
import knn.knn as knn

FOLDS = 10
points, classes = reader.read_data("svm/chips.txt")


def check_svm(fold):
    svm = SVM(C=5, gamma=1.5)

    svm.fit(fold["train_p"], fold["train_c"])
    c = fold["test_c"]
    p = svm.predict(fold["test_p"])

    return p, c


def check_knn(fold):
    k = 9
    power = 2
    metric = (lambda __x1, __x2: metrics.minkowski(__x1, __x2, power))

    ct = knn.validation(fold, metric, kernels.kernels[4], k)

    return ct["knn_c"], ct["test_c"]


def compare():
    folds = ml.folds(points, classes, FOLDS, shuffle=True)

    x = []
    y = []
    for fold in folds:
        svm_p, svm_c = check_svm(fold)
        knn_p, knn_c = check_knn(fold)

        x.extend(abs(svm_p + svm_c) / 2)
        y.extend(abs(knn_p + knn_c) / 2)

    x = np.array(x)
    y = np.array(y)

    print wilcoxon.get_good_p_value(x, y)
    print wilcoxon.get_p_value(x, y)


compare()
