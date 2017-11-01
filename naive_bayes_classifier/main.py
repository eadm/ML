import ml
import spam_detection as sd
import reader

from model.message import MessageType

cv = ml.create_cv_from_blocks(reader.read_blocks("pu1"))
__dict = ml.create_dict(cv[0]["train"], count_twice=False)
i = 4

answer = [0, 0]
for message in cv[0]["test"]:
    print "--------------"
    p = sd.get_message_spam_probability(__dict, message)
    if ((p > 0.5) & (message.type == 1)) | ((p <= 0.5) & (message.type == 2)):
        answer[0] += 1
    else:
        answer[1] += 1

print "correct: " + str(answer[0]) + ", incorrect: " + str(answer[1])
