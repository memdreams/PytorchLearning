import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data


def PCA(x):
    xm = torch.mean(x, 1, keepdim=True)
    xc = torch.matmul(xm, torch.transpose(xm, 0, -1))
    es, vs = torch.symeig(xc, eigenvectors=True)
    return es, vs

x = Variable(torch.randn(5,3), requires_grad=True)
y = Variable(torch.randn(5,2))

xx = Variable(torch.ones((2,3)), requires_grad=True)
xx.data[0][0] = 3
yy = torch.mean(xx)
z = 2*xx
#Why z.backward() is wrong?!

linear = nn.Linear(3, 2)
print('w:', linear.weight)  # random initial data from -1 to 1
print('b:', linear.bias)

# build loss and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward propagation
pred = linear(x)
lr = 0.01

loss = loss_function(pred, y)
print('loss:', loss.data[0])

# backpropagation
loss.backward()

out = z.sum()
out.backward() # out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
x.grad #grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step optimization(gradient descent)
optimizer.step()

# or do it by yourself
# linear.weight.data.sub_(lr * linear.weight.grad.data)
# linear.bias.data.sub_(lr * linear.bias.grad.data)

pred = linear(x)
loss = loss_function(pred, y)
print("The loss after once optimizing: ", loss.data[0])


# Implementing the input pipeline
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)
test_dataset = dsets.CIFAR10(root='./data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)
image, label = train_dataset[0]
print(image.size())
print(label)

# Data loader
train_loader = data.DataLoader(train_dataset,
                               batch_size=100,
                               shuffle=True,
                               num_workers=2)

test_loader = data.DataLoader(test_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
show((image+1)/2).resize((100,100))

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4, 2, 10, 3

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

b1 = torch.zeros((N, H))

h = x.mm(w1) + b1
print(h)





# x_np = np.array([[1,2,3], [4,5,6]], dtype=float)
# x = torch.from_numpy(x_np)
# es, vs = PCA(x)


