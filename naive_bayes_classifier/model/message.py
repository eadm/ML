class MessageType:
    def __init__(self):
        pass

    SPAM = 0
    LEGIT = 1


class Message:
    def __init__(self, subject, text, message_type):
        self.subject = subject
        self.text = text
        self.type = message_type

    def __str__(self):
        return "Message(subject=" + str(self.subject) + ", text=" + str(self.text) + ", type=" + str(self.type) + ")"
