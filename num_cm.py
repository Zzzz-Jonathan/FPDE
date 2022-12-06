import torch
import numpy as np
from torch.utils.data import DataLoader
from parameter import sparse_cm, LOSS, device, gradients, sparse_init


def gauss_kernel(size=5, sigma=1):
    x = np.linspace(-3 * sigma, 3 * sigma, size)

    kernel = 1 / (2 * np.pi * sigma ** 2) ** (1 / 2) * np.exp(- (x ** 2) / (2 * sigma ** 2))
    w = kernel.sum()  # 计算归一化系数
    kernel = (1 / w) * kernel

    return kernel


def dynamic_k(n):
    k = []
    n = torch.round(n * data_std[2] + data_mean[2])

    for i in n:
        if i == 14:
            k.append([530.39, 0.066, -46.42])

        elif i == 16:
            k.append([484.74, 0.065, -43.15])

        elif i == 18:
            k.append([636.68, 0.070, -45.48])

        elif i == 20:
            k.append([982.26, 0.078, -47.65])

    k = torch.FloatTensor(k).to(device)

    k1, k2, k3 = torch.split(k, 1, dim=1)

    return k1, k2, k3


def loss_pde(nn, data_inp):
    [t, x, n] = torch.split(data_inp, 1, dim=1)
    data_inp = torch.cat([t, x, n], dim=1)

    k1, k2, k3 = dynamic_k(n)

    c = nn(data_inp) * label_std + label_mean

    zeros = torch.zeros_like(c)

    c_t = gradients(c, t) / data_std[0]

    c_xx = gradients(c, x, order=2) / (data_std[1] ** 2)

    l = c_t - (k1 * c_xx + k2 * c + k3 * (c ** 2))

    return LOSS(l, zeros)


def loss_fpde(nn, data_inp, dx=2):
    pass
    [t, x, n] = torch.split(data_inp, 1, dim=1)

    k1, k2, k3 = dynamic_k(n)
    dx /= data_std[1]

    xxs = [x, x + dx, x - dx, x + 2 * dx, x - 2 * dx]

    txns = []

    for xs in xxs:
        txns.append(torch.cat([t, xs, n], dim=1))

    nn_outs = []

    for txy, weight in zip(txns, [0, 1, 1, 2, 2]):
        out = nn(txy) * label_std + label_mean

        w = kernel[2 + weight]
        nn_outs.append(w * out)

    c = sum(nn_outs)
    zeros = torch.zeros_like(c)

    c_t = gradients(c, t) / data_std[0]

    c_xx = gradients(c, x, order=2) / (data_std[1] ** 2)

    l = c_t - (k1 * c_xx + k2 * c + k3 * (c ** 2))

    return LOSS(l, zeros)


def loss_data(nn, x, y):
    y = y * label_std + label_mean

    return LOSS(nn(x) * label_std + label_mean, y)


def loss_neumann_bc(nn, data_inp):
    [t, x, n] = torch.split(data_inp, 1, dim=1)
    data_inp = torch.cat([t, x, n], dim=1)

    c = nn(data_inp) * label_std + label_mean
    zeros = torch.zeros_like(c)

    c_x = gradients(c, x) / data_std[1]

    return LOSS(c_x, zeros)


kernel = gauss_kernel(5)

data = np.load('data/cell_migration/data.npy')
label = np.load('data/cell_migration/label.npy')
bc_data = np.load('data/cell_migration/data_bc.npy')
ic_data = np.load('data/cell_migration/data_ic.npy')
ic_label = np.load('data/cell_migration/label_ic.npy')

data = data[[0, 3]]
label = label[[0, 3]]

data = data.reshape(-1, 3)
label = label.reshape(-1, 1)
bc_data = bc_data.reshape(-1, 3)
ic_data = ic_data.reshape(-1, 3)
ic_label = ic_label.reshape(-1, 1)

data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
label_mean, label_std = np.mean(label, axis=0), np.std(label, axis=0)

data = (data - data_mean) / data_std
ic_data = (ic_data - data_mean) / data_std
bc_data = (bc_data - data_mean) / data_std
label = (label - label_mean) / label_std
ic_label = (ic_label - label_mean) / label_std

rng_state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(rng_state)
np.random.shuffle(label)

np.random.set_state(rng_state)
np.random.shuffle(ic_data)
np.random.set_state(rng_state)
np.random.shuffle(ic_label)

train_len = int(data.shape[0] / 2 ** sparse_cm)
init_len = int(ic_data.shape[0] / 2 ** sparse_init)

full_data = torch.FloatTensor(data).requires_grad_(True).to(device)
train_data = torch.FloatTensor(data[:train_len]).requires_grad_(True).to(device)
train_label = torch.FloatTensor(label[:train_len]).requires_grad_(True).to(device)
bc_data = torch.FloatTensor(bc_data).requires_grad_(True).to(device)
val_data = torch.FloatTensor(data[train_len:]).requires_grad_(True).to(device)
val_label = torch.FloatTensor(label[train_len:]).requires_grad_(True).to(device)
ic_data = torch.FloatTensor(ic_data[:init_len]).requires_grad_(True).to(device)
ic_label = torch.FloatTensor(ic_label[:init_len]).requires_grad_(True).to(device)

train_data = torch.cat([train_data, ic_data], dim=0)
train_label = torch.cat([train_label, ic_label], dim=0)

data_mean, data_std = torch.FloatTensor(data_mean).to(device), torch.FloatTensor(data_std).to(device)
label_mean, label_std = torch.FloatTensor(label_mean).to(device), torch.FloatTensor(label_std).to(device)

dataset = (train_data, train_label, val_data, val_label, bc_data)

print(49.64 * 1e-3)

