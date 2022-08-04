import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN_SIZE = [3] + 5 * [2 * 50] + [3]
NN_SIZE_3D = [5] + 5 * [2 * 50] + [4]
Re = 200
Pe = 10 * Re
BATCH = 512
module_name = 'Cylinder'
EPOCH = 50000
ITERATION = 50000
collocation_size = 2 ** 14
sparse_num = 10
train_size_rate = 1 / (2 ** sparse_num)
noisy_num = 6
noisy_rate = noisy_num / 4
PICK = 0
BATCH_3d = 16
size_3d_rate = [1 / i for i in range(10, 100, 2)]
noisy_3d_rate = 1 / 2

LOSS = torch.nn.MSELoss().to(device)