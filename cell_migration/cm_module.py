import torch

"""
This file defines a NN model of cell migration. 

The input dimensions of this model are 3, which are time t, position x and initial cell number n. 
The model will only output cell number C for given t,x,n.
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


if __name__ == '__main__':
    # NN = Module([3] + 10 * [4 * 50] + [3])

    # print('parameters:', sum(param.numel() for param in NN.parameters()) / 1e6)

    pass
