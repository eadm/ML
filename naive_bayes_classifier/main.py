import ml
import spam_detection as sd
import reader

from model.message import MessageType

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))
__dict = ml.create_dict(cv[0]["test"], count_twice=False)
i = 4

print sd.get_message_spam_probability(__dict, cv[0]["test"][i])
print cv[0]["test"][i].type
