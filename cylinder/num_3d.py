from parameter import train_size_rate, noisy_rate
import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from parameter import device, BATCH, Re, LOSS, PICK

"""
This file defines the calculation methods of PDE/FPDE loss, data loss and icbc loss.
Physical constraints are specifically defined based on the given 2d N-S equations. 
For the definition method, refer to the function loss_pde. 
For the FPDE definition method of this equation, refer to the function loss_fpde.

At the same time, the file also processes the structured data and creates a dataset object.
This document defines both sparse data sets and noisy data sets. 
The parameter defined in parameter.py:
    train_size_rate --------   determines the sampling ratio of sparse data sets.
    noise_rate      --------   determines the intensity of artificial noise.
Since the efficiency of FPDE and traditional PINN is inconsistent, the batches of their data sets are also different.    
"""

data_dirs = ['data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat',
             'data/Cylinder2D.mat',
             'data/Stenosis2D.mat']

numerical_data = scipy.io.loadmat(data_dirs[PICK])
lab = '_data' if PICK == 0 else '_star'
# 'data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat'

k_type = 2
if k_type == 0:
    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
elif k_type == 1:
    kernel = [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
              [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
              [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
              [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
              [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]]
elif k_type == 2:
    kernel = [[0.057765239856828944, 0.09615773797943991, -3.8981718325193755e-17, 0.09615773797943991, 0.057765239856828944],
              [0.09615773797943991, -0.21695429437747635, 3.8981718325193755e-17, -0.21695429437747635, 0.09615773797943991],
              [-3.8981718325193755e-17, 3.8981718325193755e-17, 1, 3.8981718325193755e-17, -3.8981718325193755e-17],
              [0.09615773797943991, -0.21695429437747635, 3.8981718325193755e-17, -0.21695429437747635, 0.09615773797943991],
              [0.057765239856828944, 0.09615773797943991, -3.8981718325193755e-17, 0.09615773797943991, 0.057765239856828944]]


xs_idx = [0, 1, 1, 2, 2]
ys_idx = [0, 1, 1, 2, 2]
weights = [[i, j] for i in xs_idx for j in ys_idx]


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


def loss_pde(nn, data_inp):
    [t, x, y] = torch.split(data_inp, 1, dim=1)
    data_inp = torch.cat([t, x, y], dim=1)

    out = nn(data_inp.to(device))

    [u, v, p] = torch.split(out, 1, dim=1)
    zeros = torch.zeros_like(u)

    # c_t = gradients(c, t)
    u_t = gradients(u, t)
    v_t = gradients(v, t)

    # c_x = gradients(c, x)
    u_x = gradients(u, x)
    v_x = gradients(v, x)

    # c_y = gradients(c, y)
    u_y = gradients(u, y)
    v_y = gradients(v, y)

    # c_xx = gradients(c_x, x)
    u_xx = gradients(u_x, x)
    v_xx = gradients(v_x, x)

    # c_yy = gradients(c_y, y)
    u_yy = gradients(u_y, y)
    v_yy = gradients(v_y, y)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    # l1 = LOSS(c_t + (u * c_x + v * c_y) - (1.0 / Pe) * (c_xx + c_yy), zeros)
    l2 = LOSS(u_t + (u * u_x + v * u_y) + p_x - (1.0 / Re) * (u_xx + u_yy), zeros)
    l3 = LOSS(v_t + (u * v_x + v * v_y) + p_y - (1.0 / Re) * (v_xx + v_yy), zeros)
    l4 = LOSS(u_x + v_y, zeros)

    return l2, l3, l4


def loss_data(nn, data_inp, label):
    out = nn(data_inp.to(device))

    # [c_label, _, _, _] = torch.split(label, 1, dim=1)
    # [c_out, _, _, _] = torch.split(out, 1, dim=1)

    return LOSS(out, label), LOSS(out.std(axis=0), label.std(axis=0))


def loss_icbc(nn, size=3000):
    rng_state = np.random.get_state()
    np.random.shuffle(ic_u_v_p)
    np.random.set_state(rng_state)
    np.random.shuffle(ic_t_x_y)
    np.random.set_state(rng_state)
    np.random.shuffle(bc_t_x_y)

    ict = torch.FloatTensor(ic_t_x_y[:size]).to(device)
    icc = torch.FloatTensor(ic_u_v_p[:size, :2]).to(device)
    bct = torch.FloatTensor(bc_t_x_y[:size]).to(device)

    nn_icc = nn(ict)[:, :2]
    nn_bcc = nn(bct)[:, :2]

    bcc = torch.zeros_like(nn_bcc)

    return LOSS(nn_icc, icc), LOSS(nn_bcc, bcc)


def loss_collcation(nn, size, method):
    np.random.shuffle(collcation_points)
    points = torch.FloatTensor(collcation_points[:size]).requires_grad_(True).to(device)

    if method == 'les':
        return loss_fpde(nn, points)
    elif method == 'ns':
        return loss_pde(nn, points)


def loss_fpde(nn, data_inp, dx=0.1, dy=0.1):
    [t, x, y] = torch.split(data_inp, 1, dim=1)
    # data_inp = torch.cat([t, x, y], dim=1)
    #
    # t = data_inp[:, 0, None]
    # x = data_inp[:, 1, None]
    # y = data_inp[:, 2, None]

    xxs = [x, x + dx, x - dx, x + 2 * dx, x - 2 * dx]
    yys = [y, y + dy, y - dy, y + 2 * dy, y - 2 * dy]

    # xs_idx = [0, 1, 1, 2, 2]
    # ys_idx = [0, 1, 1, 2, 2]
    # weight_idx = [i + j for i in xs_idx for j in ys_idx]

    txys = []

    for xs in xxs:
        for ys in yys:
            txys.append(torch.cat([t, xs, ys], dim=1))

    nn_outs = []
    nn_uvs = []
    nn_uus = []
    nn_vvs = []
    # nn_ucs = []
    # nn_vcs = []

    for txy, weight in zip(txys, weights):
        out = nn(txy.to(device))
        w = kernel[2 + weight[0]][2 + weight[1]]
        nn_outs.append(w * out)
        nn_uvs.append(w * out[:, 0] * out[:, 1])
        nn_uus.append(w * out[:, 0] * out[:, 0])
        nn_vvs.append(w * out[:, 1] * out[:, 1])
        # nn_ucs.append(w * out[:, 0] * out[:, 1])
        # nn_vcs.append(w * out[:, 0] * out[:, 2])

    out_bar = sum(nn_outs)
    uv_bar = sum(nn_uvs)
    uu_bar = sum(nn_uus)
    vv_bar = sum(nn_vvs)
    # uc_bar = sum(nn_ucs)
    # vc_bar = sum(nn_vcs)

    [u_bar, v_bar, p_bar] = torch.split(out_bar, 1, dim=1)

    # c_bar_t = gradients(c_bar, t)
    u_bar_t = gradients(u_bar, t)
    v_bar_t = gradients(v_bar, t)

    uu_bar_x = gradients(uu_bar, x)
    uv_bar_x = gradients(uv_bar, x)

    uv_bar_y = gradients(uv_bar, y)
    vv_bar_y = gradients(vv_bar, y)

    # uc_bar_x = gradients(uc_bar, x)
    # vc_bar_y = gradients(vc_bar, y)

    p_bar_x = gradients(p_bar, x)
    p_bar_y = gradients(p_bar, y)

    # c_bar_x = gradients(c_bar, x)
    # c_bar_y = gradients(c_bar, y)

    u_bar_x = gradients(u_bar, x)
    v_bar_y = gradients(v_bar, y)

    # c_bar_xx = gradients(c_bar_x, x)
    # c_bar_yy = gradients(c_bar_y, y)

    u_bar_xx = gradients(u_bar_x, x)
    u_bar_yy = gradients(u_bar, y, order=2)

    v_bar_xx = gradients(v_bar, x, order=2)
    v_bar_yy = gradients(v_bar_y, y)

    zeros = torch.zeros_like(u_bar)

    # l1 = LOSS(c_bar_t + (uc_bar_x + vc_bar_y) - (1.0 / Pe) * (c_bar_xx + c_bar_yy), zeros)
    l2 = LOSS(u_bar_t + (uu_bar_x + uv_bar_y) + p_bar_x - (1 / Re) * (u_bar_yy + u_bar_xx), zeros)
    l3 = LOSS(v_bar_t + (uv_bar_x + vv_bar_y) + p_bar_y - (1 / Re) * (v_bar_xx + v_bar_yy), zeros)
    l4 = LOSS(u_bar_x + v_bar_y, zeros)

    return l2, l3, l4

    # print(gradients(p_bar, x))

    # out = u(txys[1].to(device))
    # [_, _, _, pp] = torch.split(out, 1, dim=1)

    # print(gradients(pp, yys[0]))
    # print(gradients(pp, yys[1])) # same

    # for txy, weight in zip(txys, weights):
    #     out = u(txy.to(device))
    #     [_, _, _, pp] = torch.split(out, 1, dim=1)
    #     print(kernel[2 + weight[0]][2 + weight[1]] * gradients(pp, x))


t_star = numerical_data['t' + lab]  # T x 1, [0 -> 16]
x_star = numerical_data['x' + lab]  # N x 1
y_star = numerical_data['y' + lab]  # N x 1
U_star = numerical_data['U' + lab][:, :, None]  # N x T
V_star = numerical_data['V' + lab][:, :, None]  # N x T
P_star = numerical_data['P' + lab][:, :, None]  # N x T
# C_star = numerical_data['C' + lab][:, :, None]  # N x T

# print(np.max(P_star), np.min(P_star))

N, T = U_star.shape[0], U_star.shape[1]

t_x_y = np.zeros((N, T, 3))
for t in range(len(t_star)):
    t_x_y[:, t, 0] = t_star[t][0]

x_y = np.concatenate((x_star, y_star), axis=1)
for xy in range(len(x_y)):
    t_x_y[xy, :, 1] = x_y[xy][0]
    t_x_y[xy, :, 2] = x_y[xy][1]

u_v_p = np.concatenate((U_star, V_star, P_star), axis=2)
x_min = x_star.min()

t_x_y = t_x_y.reshape((N * T, 3))
u_v_p = u_v_p.reshape(N * T, 3)
# print(t_x_y.shape)

ic_t_x_y = t_x_y[t_x_y[:, 1] == x_min]
ic_u_v_p = u_v_p[t_x_y[:, 1] == x_min]
bc_t_x_y = np.array(
    [[t, np.cos(i) / 2, np.sin(i) / 2] for i in np.linspace(0, 2 * np.pi, 200) for t in np.linspace(0, 16, 200)])

collcation_points = np.array(
    [[t, x, y] for t in np.arange(0, 16, 0.5) for x in np.arange(-2.5, 7.5, 0.4) for y in np.arange(-2.5, 2.5, 0.4)])
# data = dataset(torch.tensor(t_x_y).requires_grad_(True).type(torch.float32), torch.tensor(c_u_v_p).type(torch.float32))
#
# dataloader = DataLoader(dataset=data,
#                         batch_size=BATCH,
#                         shuffle=True,
#                         num_workers=0)

rng_state = np.random.get_state()
np.random.shuffle(t_x_y)
np.random.set_state(rng_state)
np.random.shuffle(u_v_p)

var_cuvp = np.var(u_v_p, axis=0)
std_cuvp = noisy_rate * np.sqrt(var_cuvp)

norm_size = int(len(t_x_y) / 2)
rare_size = int(len(t_x_y) * train_size_rate)
# print(rare_size)

rare_data = t_x_y[:rare_size]
rare_label = u_v_p[:rare_size]
noisy_rare_label = np.random.normal(rare_label, std_cuvp)

validation_data = torch.FloatTensor(t_x_y[norm_size:norm_size + 10000]).requires_grad_(False).to(device)
validation_label = torch.FloatTensor(u_v_p[norm_size:norm_size + 10000]).requires_grad_(False).to(device)

# collocation_points = np.concatenate([16 * np.random.rand(collocation_size, 1),
#                                      10 * np.random.rand(collocation_size, 1) - 2.5,
#                                      5 * np.random.rand(collocation_size, 1) - 2.5], axis=1)
# collocation_points = torch.FloatTensor(collocation_points).requires_grad_(True).to(device)

rare_dataset = dataset(torch.FloatTensor(rare_data),
                       torch.FloatTensor(rare_label))
rare_dataloader_1 = DataLoader(dataset=rare_dataset,
                               batch_size=BATCH,
                               shuffle=True,
                               num_workers=8)
rare_dataloader_2 = DataLoader(dataset=rare_dataset,
                               batch_size=25 * BATCH,
                               shuffle=True,
                               num_workers=8)

noisy_rare_dataset = dataset(torch.FloatTensor(rare_data),
                             torch.FloatTensor(noisy_rare_label))
noisy_rare_dataloader_1 = DataLoader(dataset=noisy_rare_dataset,
                                     batch_size=BATCH,
                                     shuffle=True,
                                     num_workers=8)
noisy_rare_dataloader_2 = DataLoader(dataset=noisy_rare_dataset,
                                     batch_size=25 * BATCH,
                                     shuffle=True,
                                     num_workers=8)

if __name__ == '__main__':
    for x, y in rare_dataloader_2:
        print(x.shape, y.shape)