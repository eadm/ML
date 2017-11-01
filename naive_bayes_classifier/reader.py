from os import listdir
from model.message import Message, MessageType


def __message_type_from_filename(filename):
    if "spmsg" in filename:
        return MessageType.SPAM
    else:
        return MessageType.LEGIT


def read_block(path):
    __block = []
    for filename in listdir(path):
        with open(path + filename) as f:
            lines = f.readlines()
            __subject = map(int, lines[0].strip().split(' ')[1:])
            __text = map(int, lines[2].strip().split(' '))
            __type = __message_type_from_filename(filename)
            __block.append(Message(__subject, __text, __type))
    return __block


block = read_block("pu1/part1/")


