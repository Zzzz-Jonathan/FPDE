import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_SIZE = [3] + 10 * [4 * 50] + [4]
NN_SIZE_3D = [5] + 10 * [4 * 50] + [4]
Re = 200
Pe = 10 * Re
BATCH = 256
module_name = 'Cylinder_%d' % Re
EPOCH = 10000
ITERATION = 15000
collocation_size = 2 ** 14
train_size_rate = 1 / (2 ** 0)
noisy_rate = 2 / 4
PICK = 0
size_3d_rate = [1 / 700, 1 / 600, 1 / 500, 1 / 400, 1 / 300, 1 / 200, 1 / 100]

LOSS = torch.nn.MSELoss().to(device)