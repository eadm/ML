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
    spam_bound = 0.53  # 0.53 -- best

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
    print table

print np.array(f1s).mean()
