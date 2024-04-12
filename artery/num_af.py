import torch
import numpy as np
from parameter import LOSS, device, gradients

"""
This file defines the calculation methods of PDE/FPDE loss, data loss and interface loss.
Physical constraints are specifically defined based on the given blood flow equation. 
For the definition method, refer to the function loss_pde. 
For the FPDE definition method of this equation, refer to the function loss_fpde.

At the same time, the file also processes the structured data and creates a dataset object.
"""


def gauss_kernel(size=5, sigma=1):
    x = np.linspace(-3 * sigma, 3 * sigma, size)

    kernel = 1 / (2 * np.pi * sigma ** 2) ** (1 / 2) * np.exp(- (x ** 2) / (2 * sigma ** 2))
    w = kernel.sum()  # 计算归一化系数
    kernel = (1 / w) * kernel

    return kernel


def loss_pde(nn, data_inp, area):
    [t, x] = torch.split(data_inp, 1, dim=1)
    data_inp = torch.cat([t, x], dim=1)

    out = nn(data_inp, area) * label_std + label_mean
    [A, U, p] = torch.split(out, 1, dim=1)

    zeros = torch.zeros_like(A)

    p_x = gradients(p, x) / data_std[1]

    A_t = gradients(A, t) / data_std[0]
    A_x = gradients(A, x) / data_std[1]

    U_t = gradients(U, t) / data_std[0]
    U_x = gradients(U, x) / data_std[1]

    l1 = LOSS(A_t + U * A_x + A * U_x, zeros)
    l2 = LOSS(U_t + p_x + U * U_x, zeros)

    return l1 + l2


def loss_interface(nn, data_interface):
    out1 = nn(data_interface, 1) * label_std + label_mean
    [A1, U1, p1] = torch.split(out1, 1, dim=1)
    out2 = nn(data_interface, 2) * label_std + label_mean
    [A2, U2, p2] = torch.split(out2, 1, dim=1)
    out3 = nn(data_interface, 3) * label_std + label_mean
    [A3, U3, p3] = torch.split(out3, 1, dim=1)

    zeros = torch.zeros_like(A1)

    l1 = LOSS(A1 * U1 - A2 * U2 - A3 * U3, zeros)
    l2 = LOSS(p1 + (U1 ** 2) / 2, p2 + (U2 ** 2) / 2)
    l3 = LOSS(p1 + (U1 ** 2) / 2, p3 + (U3 ** 2) / 2)

    return l1 + l2 + l3


def loss_fpde(nn, data_inp, area, dx=0.01):
    [t, x] = torch.split(data_inp, 1, dim=1)

    dx /= data_std[1]

    xxs = [x, x + dx, x - dx, x + 2 * dx, x - 2 * dx]

    txs = []

    for xs in xxs:
        txs.append(torch.cat([t, xs], dim=1))

    nn_outs = []

    for txy, weight in zip(txs, [0, 1, 1, 2, 2]):
        out = nn(txy, area) * label_std + label_mean

        w = kernel[2 + weight]
        nn_outs.append(w * out)

    [A, U, p] = torch.split(sum(nn_outs), 1, dim=1)
    zeros = torch.zeros_like(A)

    p_x = gradients(p, x) / data_std[1]

    A_t = gradients(A, t) / data_std[0]
    A_x = gradients(A, x) / data_std[1]

    U_t = gradients(U, t) / data_std[0]
    U_x = gradients(U, x) / data_std[1]

    l1 = LOSS(A_t + U * A_x + A * U_x, zeros)
    l2 = LOSS(U_t + p_x + U * U_x, zeros)

    return l1 + l2


def loss_data(nn, x, y, area):
    out = nn(x, area)
    [A, U, _] = torch.split(out, 1, dim=1)
    [A_ref, U_ref] = torch.split(y, 1, dim=1)

    return LOSS(A, A_ref) + LOSS(U, U_ref)


def to_torch(x, grad=True):
    l = []
    for i in x:
        if grad:
            l.append(torch.FloatTensor(i).requires_grad_(True).to(device))
        else:
            l.append(torch.FloatTensor(i).to(device))

    return l


kernel = gauss_kernel(5)

path = '../data/arterial_flow/'
data = np.load(path + 'train_data.npy') * 1e-3
label = np.load(path + 'train_label.npy') * np.array([1e-6, 1e-2])
collocation = np.load(path + 'collocation_data.npy') * 1e-3
interface = np.load(path + 'interface_data.npy') * 1e-3
test_data = np.load(path + 'test_data.npy') * 1e-3
test_label = np.load(path + 'test_label.npy') * np.array([1e-6, 1e-2])

data_mean, data_std = np.mean(data, axis=(0, 1)), np.std(data, axis=(0, 1))
label_mean, label_std = np.mean(label, axis=(0, 1)), np.std(label, axis=(0, 1))

# data = (data - data_mean) / data_std
test_data = (test_data - data_mean) / data_std
# interface = (interface - data_mean) / data_std
# collocation = (collocation - data_mean) / data_std
#
# label = (label - label_mean) / label_std
# test_label = (test_label - label_mean) / label_std

# idx = np.random.choice(800, size=sparse_af, replace=False)
# data = data[:, idx, :]
# label = label[:, idx, :]

data = to_torch(data, False)
label = to_torch(label, False)
collocation = to_torch(collocation)
interface = to_torch([interface], False)
test_data = to_torch([test_data], False)
test_label = to_torch([test_label], False)

data_mean, data_std = torch.FloatTensor(data_mean).to(device), torch.FloatTensor(data_std).to(device)
label_mean = torch.FloatTensor(np.concatenate((label_mean, [0]))).to(device)
label_std = torch.FloatTensor(np.concatenate((label_std, [1]))).to(device)

dataset = (data, label, test_data[0], test_label[0], collocation, interface)

