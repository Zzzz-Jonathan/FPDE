import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_SIZE = [3] + 10 * [4 * 50] + [4]
Re = 200
Pe = 10 * Re
BATCH = 1024
module_name = 'Cylinder_%d' % Re
EPOCH = 10000

LOSS = torch.nn.MSELoss().to(device)