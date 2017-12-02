import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
def PCA(x):
    xm = torch.mean(x, 1, keepdim=True)
    xc = torch.matmul(xm, torch.transpose(xm, 0, -1))
    es, vs = torch.symeig(xc, eigenvectors=True)
    return es, vs

x = Variable(torch.randn(5,3))
y = Variable(torch.randn(5,2))

linear = nn.Linear(3, 2)
print('w:', linear.weight)  # random initial data from -1 to 1
print('b:', linear.bias)




# x_np = np.array([[1,2,3], [4,5,6]], dtype=float)
# x = torch.from_numpy(x_np)
# es, vs = PCA(x)


