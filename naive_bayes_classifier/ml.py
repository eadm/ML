import numpy as np


def create_cv_from_blocks(blocks):
    folds = []
    for i in range(len(blocks)):
        train = np.concatenate(blocks[:i] + blocks[i + 1:])
        test = blocks[i]
        folds.append({"train": train, "test": test})

    return folds

