import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from parameter import device, BATCH, Re, Pe, LOSS, PICK

data_dirs = ['data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat',
             'data/Cylinder2D.mat',
             'data/Stenosis2D.mat']

numerical_data = scipy.io.loadmat(data_dirs[PICK])
lab = '_data' if PICK == 0 else '_star'
# 'data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat'

kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
          [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
          [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
          [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
          [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
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

    [c, u, v, p] = torch.split(out, 1, dim=1)
    zeros = torch.zeros_like(u)

    c_t = gradients(c, t)
    u_t = gradients(u, t)
    v_t = gradients(v, t)

    c_x = gradients(c, x)
    u_x = gradients(u, x)
    v_x = gradients(v, x)

    c_y = gradients(c, y)
    u_y = gradients(u, y)
    v_y = gradients(v, y)

    c_xx = gradients(c_x, x)
    u_xx = gradients(u_x, x)
    v_xx = gradients(v_x, x)

    c_yy = gradients(c_y, y)
    u_yy = gradients(u_y, y)
    v_yy = gradients(v_y, y)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    l1 = LOSS(c_t + (u * c_x + v * c_y) - (1.0 / Pe) * (c_xx + c_yy), zeros)
    l2 = LOSS(u_t + (u * u_x + v * u_y) + p_x - (1.0 / Re) * (u_xx + u_yy), zeros)
    l3 = LOSS(v_t + (u * v_x + v * v_y) + p_y - (1.0 / Re) * (v_xx + v_yy), zeros)
    l4 = LOSS(u_x + v_y, zeros)

    return l1, l2, l3, l4


def loss_data(nn, data_inp, label):
    out = nn(data_inp.to(device))

    # [c_label, _, _, _] = torch.split(label, 1, dim=1)
    # [c_out, _, _, _] = torch.split(out, 1, dim=1)

    return LOSS(out, label)


def loss_les(nn, data_inp, dx=0.1, dy=0.1):
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
    nn_ucs = []
    nn_vcs = []

    for txy, weight in zip(txys, weights):
        out = nn(txy.to(device))
        w = kernel[2 + weight[0]][2 + weight[1]]
        nn_outs.append(w * out)
        nn_uvs.append(w * out[:, 1] * out[:, 2])
        nn_uus.append(w * out[:, 1] * out[:, 1])
        nn_vvs.append(w * out[:, 2] * out[:, 2])
        nn_ucs.append(w * out[:, 0] * out[:, 1])
        nn_vcs.append(w * out[:, 0] * out[:, 2])

    out_bar = sum(nn_outs)
    uv_bar = sum(nn_uvs)
    uu_bar = sum(nn_uus)
    vv_bar = sum(nn_vvs)
    uc_bar = sum(nn_ucs)
    vc_bar = sum(nn_vcs)

    [c_bar, u_bar, v_bar, p_bar] = torch.split(out_bar, 1, dim=1)

    c_bar_t = gradients(c_bar, t)
    u_bar_t = gradients(u_bar, t)
    v_bar_t = gradients(v_bar, t)

    uu_bar_x = gradients(uu_bar, x)
    uv_bar_x = gradients(uv_bar, x)

    uv_bar_y = gradients(uv_bar, y)
    vv_bar_y = gradients(vv_bar, y)

    uc_bar_x = gradients(uc_bar, x)
    vc_bar_y = gradients(vc_bar, y)

    p_bar_x = gradients(p_bar, x)
    p_bar_y = gradients(p_bar, y)

    c_bar_x = gradients(c_bar, x)
    c_bar_y = gradients(c_bar, y)

    u_bar_x = gradients(u_bar, x)
    v_bar_y = gradients(v_bar, y)

    c_bar_xx = gradients(c_bar_x, x)
    c_bar_yy = gradients(c_bar_y, y)

    u_bar_xx = gradients(u_bar_x, x)
    u_bar_yy = gradients(u_bar, y, order=2)

    v_bar_xx = gradients(v_bar, x, order=2)
    v_bar_yy = gradients(v_bar_y, y)

    zeros = torch.zeros_like(u_bar)

    l1 = LOSS(c_bar_t + (uc_bar_x + vc_bar_y) - (1.0 / Pe) * (c_bar_xx + c_bar_yy), zeros)
    l2 = LOSS(u_bar_t + (uu_bar_x + uv_bar_y) + p_bar_x - (1 / Re) * (u_bar_yy + u_bar_xx), zeros)
    l3 = LOSS(v_bar_t + (uv_bar_x + vv_bar_y) + p_bar_y - (1 / Re) * (v_bar_xx + v_bar_yy), zeros)
    l4 = LOSS(u_bar_x + v_bar_y, zeros)

    return l1, l2, l3, l4

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
C_star = numerical_data['C' + lab][:, :, None]  # N x T

N, T = U_star.shape[0], U_star.shape[1]

t_x_y = np.zeros((N, T, 3))
for t in range(len(t_star)):
    t_x_y[:, t, 0] = t_star[t][0]

x_y = np.concatenate((x_star, y_star), axis=1)
for xy in range(len(x_y)):
    t_x_y[xy, :, 1] = x_y[xy][0]
    t_x_y[xy, :, 2] = x_y[xy][1]

c_u_v_p = np.concatenate((C_star, U_star, V_star, P_star), axis=2)

t_x_y = t_x_y.reshape((N * T, 3))
c_u_v_p = c_u_v_p.reshape(N * T, 4)

data = dataset(torch.tensor(t_x_y).requires_grad_(True).type(torch.float32), torch.tensor(c_u_v_p).type(torch.float32))

dataloader = DataLoader(dataset=data,
                        batch_size=BATCH,
                        shuffle=True,
                        num_workers=0)

if __name__ == '__main__':
    xs_idx = [[1, 1, 1],
              [2, 2, 2],
              [1, 1, 1], ]

    weight_idx = [[i, j] for i in xs_idx for j in ys_idx]

    print(weight_idx)
