import torch

from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data = data_tensor
        self.target = target_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NN_SIZE = [3] + 5 * [2 * 50] + [3]
NN_SIZE_3D = [5] + 5 * [2 * 50] + [4]

Re = 200
Pe = 10 * Re
BATCH = 128
module_name = 'Cylinder'

EPOCH = 50000
ITERATION = 50000
LR = 5e-3

collocation_size = 2 ** 14
sparse_num = 10
train_size_rate = 1 / (2 ** sparse_num)
noisy_num = 6
noisy_rate = noisy_num / 4
PICK = 0

BATCH_3d = 512
size_3d_rate = [1 / i for i in range(10, 100, 2)]
noisy_3d_rate = 1 / 2

Re_4d = 16384

LOSS = torch.nn.MSELoss().to(device)
