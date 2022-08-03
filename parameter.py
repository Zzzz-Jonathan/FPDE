import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_SIZE = [3] + 5 * [2 * 50] + [3]
NN_SIZE_3D = [5] + 5 * [2 * 50] + [4]
Re = 200
Pe = 10 * Re
BATCH = 512
module_name = 'Cylinder'
EPOCH = 35000
ITERATION = 35000
collocation_size = 2 ** 14
sparse_num = 0
train_size_rate = 1 / (2 ** sparse_num)
noisy_rate = 2 / 4
PICK = 0
size_3d_rate = [1 / i for i in range(10, 100, 2)]
noisy_3d_rate = 1 / 2

LOSS = torch.nn.MSELoss().to(device)