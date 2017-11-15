import ml
import spam_detection as sd
import numpy as np
import reader
import pylab as pl

from model.message import MessageType

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))


def ccv(spam_bound):
    f1s = []
    x = []
    y = []
    for fold in cv:
        __dict = ml.create_dict(fold["train"], count_twice=False)

        orig = []
        test = []
        # spam_bound = 0.53  # 0.53 -- best

        for message in fold["test"]:
            h = spam_bound / (1 - spam_bound)
            p = sd.get_message_spam_probability(__dict, __dict, h, h, 0.8, message)

            orig.append(message.type)
            if p > spam_bound:
                test.append(MessageType.SPAM)
            else:
                test.append(MessageType.LEGIT)

        table = ml.contingency(orig, test)
        f1s.append(table["F1"])
        x.append(table["SPE"])
        y.append(table["SEN"])
        print table

    print np.array(f1s).mean()
    return 1. - np.array(x).mean(), np.array(y).mean()


# ccv(0.53)
xs, ys = [], []
for i in np.arange(0.55, 0.85, 0.01):
    print "TR: " + str(i)
    x, y = ccv(i)
    xs.append(x)
    ys.append(y)

pl.plot(xs, ys)
pl.show()
