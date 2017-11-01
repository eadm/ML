import ml
import spam_detection as sd
import numpy as np
import reader

from model.message import MessageType

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))

f1s = []
for fold in cv:
    __dict = ml.create_dict(fold["train"], count_twice=False)

    orig = []
    test = []
    for message in fold["test"]:
        p = sd.get_message_spam_probability(__dict, message)
        orig.append(message.type)
        if p > 0.5:
            test.append(MessageType.SPAM)
        else:
            test.append(MessageType.LEGIT)

    table = ml.contingency(orig, test)
    f1s.append(table["F1"])
    print table

print np.array(f1s).mean()
