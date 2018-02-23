#https://github.com/chenyuntc/pytorch-book/blob/master/chapter3-Tensor%E5%92%8Cautograd/Tensor.ipynb
# 2018.02.21

import torch


# 指定tensor的形状
a = torch.Tensor(2, 3)
print(a[0:1, :2]) # 第0行，前两列  [torch.FloatTensor of size 1x2]
print(a[0, :2]) # 注意两者的区别：形状不同 [torch.FloatTensor of size 2]
a # 数值取决于内存空间的状态
a>1  #
a[a>1] # 等价于a.masked_select(a>1)
# 选择结果与原tensor不共享内存空间

a = torch.arange(0, 16).view(4, 4)
# 取a中的每一个元素与3相比较大的一个 (小于3的截断成3)
torch.clamp(a, min=3)
# 选取对角线的元素
index = torch.LongTensor([[0,1,2,3]])
a.gather(0, index)  # size 1x4
# 选取反对角线上的元素
index = torch.LongTensor([[3,2,1,0]]).t()
a.gather(1, index)  # size 4x1
torch.mode(a)
# a[[2, 1, 0], [0], [1]] # 等价于 a[2,0,1],a[1,0,1],a[0,0,1]
# 把a转成FloatTensor，等价于b=a.type(t.FloatTensor)
b = a.float()

b = torch.Tensor((2, 3)) # size 2, 2 and 3
x = torch.arange(1, 12, 2) # 1,3,5,7,9,11 [torch.FloatTensor of size 5]
y = x.view(2, 3) # equals x.view(-1, 3)
y.resize_(1, 3) # 1,3,5
y.unsqueeze(1) # 注意形状，在第1维（下标从0开始）上增加“１”
y.squeeze(0) # 压缩第0维的“１”
y.squeeze() # 把所有维度为“1”的压缩
x = torch.linspace(1, 10, 5) # 1, 3.25, 5.5, 7.75, 10 [torch.FloatTensor of size 5]
# tensor.shape等价于tensor.size()
x.numel()  # 5
y = x.tolist() # [1.0, 3.25, 5.5, 7.75, 10.0]
x = torch.randperm(5)
print(y)


# A Real Example
# %matplotlib inline # only used for jupyter notebook
from matplotlib import pyplot as plt
torch.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = torch.rand(batch_size, 1) * 20 #rand: 0,1分布
    y = x * 2 + (1 + torch.randn(batch_size, 1)) * 3 # randn: N(0, 1)
    return x, y

# x, y = get_fake_data()
# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# plt.show()

w = torch.rand(1, 1)
b = torch.zeros(1,1)

lr = 0.001

for i in range(2000):
    x, y = get_fake_data()

    # forward: loss
    y_pred = x@w + b.expand_as(y) # x.mm(w) = x@w
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()

    # backward: gradient
    dloss = 1
    dy_pred = dloss * (y_pred - y)
    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    w.sub_(lr*dw)
    b.sub_(lr * db)

    if i % 1000 == 0:
        x = torch.arange(0, 20).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.numpy(), y.numpy())

        x1, y1 = get_fake_data(batch_size=20)
        plt.scatter(x1.squeeze().numpy(), y1.squeeze().numpy())

        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.show()
        plt.pause(0.5)

print(w.squeeze()[0], b.squeeze()[0])



