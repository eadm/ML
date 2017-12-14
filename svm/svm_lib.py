import reader
import ml
from sklearn import svm

points, classes = reader.read_data("chips.txt")

FOLDS = 10

folds = ml.folds(points, classes, FOLDS, shuffle=True)


def check(_folds):
    ans_c = []
    ans_p = []
    for fold in _folds:
        clf = svm.SVC()
        clf.fit(fold["train_p"], fold["train_c"])
        ans_c.extend(clf.predict(fold["test_p"]))
        ans_p.extend(fold["test_c"])

    ctg = ml.contingency(ans_p, ans_c)
    print ctg
    return ctg


check(folds)
# clf = o_svm.SVC()
# print clf.fit(points, classes)

# print clf.predict([[2., 2.]])
