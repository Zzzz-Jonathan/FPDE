import torch
from parameter import NN_SIZE


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
    NN = Module(NN_SIZE)
    print(NN.net)
