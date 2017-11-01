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
            __dict.setdefault(word, {"total": 0, "spam": 0, "legit": 0})

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
