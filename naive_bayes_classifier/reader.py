from os import listdir
from os.path import isdir, join
from model.message import Message, MessageType
import numpy as np


def __message_type_from_filename(filename):
    if "spmsg" in filename:
        return MessageType.SPAM
    else:
        return MessageType.LEGIT


def read_block(path):
    __block = []
    for filename in listdir(path):
        with open(join(path, filename)) as f:
            lines = f.readlines()
            __subject = np.array(map(int, lines[0].strip().split(' ')[1:]))
            __text = np.array(map(int, lines[2].strip().split(' ')))
            __type = __message_type_from_filename(filename)
            __block.append(Message(__subject, __text, __type))
    return __block


def read_blocks(path):
    __blocks = []
    dirs = [__dir_name for __dir_name in listdir(path) if isdir(join(path, __dir_name))]
    for dir_name in dirs:
        __blocks.append(read_block(join(path, dir_name)))
    return __blocks
