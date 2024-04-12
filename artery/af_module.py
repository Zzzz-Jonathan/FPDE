import torch

"""
This file defines a NN model of arterial blood flow. 

The input dimensions of this model are 2, which are time t and position x. 
At the same time, you also need to enter which segment the current artery is located in (area=1, 2 or 3). 
The model will output velocity v, area a and pressure p based on the input and the number of artery segments.
"""

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.nn.Sigmoid()(x)


def active_fun():
    return Swish()


class AF_module(torch.nn.Module):
    def __init__(self):
        super(AF_module, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
        )

        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 3),
        )
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 3),
        )
        self.net3 = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 50),
            active_fun(),
            torch.nn.Linear(50, 3),
        )

    def forward(self, x, area):
        x = self.net(x)
        if area == 1:
            return self.net1(x)
        elif area == 2:
            return self.net2(x)
        elif area == 3:
            return self.net3(x)


if __name__ == '__main__':
    # NN = Module([3] + 10 * [4 * 50] + [3])

    # print('parameters:', sum(param.numel() for param in NN.parameters()) / 1e6)

    pass
