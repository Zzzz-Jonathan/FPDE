import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_SIZE = [3] + 10 * [4 * 50] + [4]
Re = 200
Pe = 10 * Re
BATCH = 256
module_name = 'Cylinder_%d' % Re
EPOCH = 10000
collocation_size = 2 ** 14
train_size_rate = 1 / 5000
noisy_rate = 1

LOSS = torch.nn.MSELoss().to(device)