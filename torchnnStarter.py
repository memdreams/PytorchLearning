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

