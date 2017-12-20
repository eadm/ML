# import mnist
from nn import NN


nn = NN()
for i in range(28):
    nn.add_empty_layer(768, 768)

NN.from_file('dmp.txt')
