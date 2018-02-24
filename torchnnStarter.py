# https://github.com/chenyuntc/pytorch-book/blob/master/chapter4-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B7%A5%E5%85%B7%E7%AE%B1nn/chapter4.ipynb
# 2018.02.23 Jie

import torch
import torch.nn as nn
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        y = x + self.b.expand_as(x)
        return y

layer = Linear(4, 3)
inputData = Variable(torch.randn(2,4))
output = layer(inputData)
print(output)

for name, parameter in layer.named_parameters():
    print(name, parameter)


# Create MLP by using the class Linear above
class Perceptron(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Perceptron, self).__init__()
        self.l1 = Linear(n_in, n_hidden)
        self.l2 = Linear(n_hidden, n_out)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = self.l2(x)
        return x

