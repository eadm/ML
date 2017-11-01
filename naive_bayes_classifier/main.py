import ml
import spam_detection as sd
import reader

from model.message import MessageType

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))
__dict = ml.create_dict(cv[0]["train"], count_twice=False)
i = 4

for message in cv[0]["test"]:
    print "--------------"
    print sd.get_message_spam_probability(__dict, message)
    print message.type
