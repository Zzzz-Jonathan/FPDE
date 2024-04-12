import torch

"""
This file defines a NN model of cylinder flow. 

The input dimensions of this model are 3, which are time t and position x and y. 
The model will output velocity u, v, and pressure p based on the input t,x,y.
"""


class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.nn.Sigmoid()(x)


def active_fun():
    return Swish()


class Module(torch.nn.Module):
    def __init__(self, size):
        super(Module, self).__init__()
        self.net = torch.nn.Sequential()

        for i in range(len(size) - 1):
            self.net.add_module('linear_%d' % i, torch.nn.Linear(size[i], size[i + 1]))

            if i != len(size) - 2:
                self.net.add_module('activate_%d' % i, active_fun())

    def forward(self, x):
        return self.net(x)


class ResBlock(torch.nn.Module):
    def __init__(self, size=20):
        super(ResBlock, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            active_fun(),
            torch.nn.Linear(size, size),
            active_fun(),
            torch.nn.Linear(size, size),
            active_fun()
        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            active_fun()
        )

    def forward(self, x):
        out1 = self.net(x)
        out2 = self.lin(x)
        return out1 + out2


class ResLinear(torch.nn.Module):
    def __init__(self, shape=None, block=ResBlock):
        super(ResLinear, self).__init__()
        if shape is None:
            self.shape = [4, 20, 4]
        else:
            self.shape = shape

        self.block = block
        self.front = torch.nn.Linear(self.shape[0], self.shape[1])
        self.back = torch.nn.Linear(self.shape[1], self.shape[2])

        self.net1 = self.add_blocks(16)
        self.net2 = self.add_blocks(4)
        self.net3 = self.add_blocks(1)

    def add_blocks(self, length):
        blocks = []
        for i in range(length):
            blocks.append(self.block(self.shape[1]))

        return torch.nn.Sequential(*blocks)

    def forward(self, x):
        x = self.front(x)

        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)

        return self.back(x1 + x2 + x3)


if __name__ == '__main__':
    # NN = Module([3] + 10 * [4 * 50] + [3])

    # print('parameters:', sum(param.numel() for param in NN.parameters()) / 1e6)

    print(1e6)
