import os
import torch
from module import Module
from parameter import NN_SIZE, module_name
from condition import numerical_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, griddata

TIME = [0, 5, 10, 15]


def my_plot(_x, _y, _z):
    dxs = np.linspace(-2.5, 7.5, 1000)
    dys = np.linspace(-2.5, 2.5, 500)
    dxs, dys = np.meshgrid(dxs, dys)

    z_new = griddata((_x, _y), _z, (dxs, dys), method='linear')

    plt.imshow(z_new, cmap=plt.get_cmap('hot'))
    plt.colorbar()
    # plt.contourf(dxs, dys, z_new, levels=50, cmap=plt.get_cmap('Spectral'))

    plt.show()


def scat(_x, _y, _z):
    plt.scatter(_x, _y, c=_z)
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    NN = Module(NN_SIZE)
    module_name = 'train_history/Cylinder_200_02635232'
    # writer = SummaryWriter('/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history')

    if os.path.exists(module_name):
        state = torch.load(module_name)

        NN.load_state_dict(state['model'])

    t_star = numerical_data['t_data']  # T x 1, [0 -> 16]
    x_star = numerical_data['x_data']  # N x 1
    y_star = numerical_data['y_data']  # N x 1
    U_star = numerical_data['U_data'][:, :, None]  # N x T
    V_star = numerical_data['V_data'][:, :, None]  # N x T
    P_star = numerical_data['P_data'][:, :, None]  # N x T
    C_star = numerical_data['C_data'][:, :, None]  # N x T

    N, T = U_star.shape[0], U_star.shape[1]
    x, y = x_star[:, 0], y_star[:, 0]

    for idx in np.arange(0, 16, 1):  # idx in np.arange(int(T / 10)) * 10:
        # t = t_star[idx][0]
        t = idx
        t_x_y = np.zeros((N, 3))

        t_x_y[:, 0] = t
        t_x_y[:, 1] = x
        t_x_y[:, 2] = y

        # x_y = np.concatenate((x_star, y_star), axis=1)
        # c = C_star[:, idx, 0]
        #
        # my_plot(x, y, c)

        out = NN(torch.FloatTensor(t_x_y)).detach().numpy()
        c_NN = out[:, 3]

        my_plot(x, y, c_NN)
