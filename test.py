import os
import torch
from module import Module
from parameter import NN_SIZE, module_name, noisy_rate
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

    cylinder = plt.Circle(xy=(250, 250), radius=50, alpha=1, color='white')
    plt.gca().add_patch(cylinder)

    plt.show()


def scat(_x, _y, _z):
    plt.scatter(_x, _y, c=_z)
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    NN2 = Module(NN_SIZE)
    module_name = 'Cylinder_200_les_ns'

    if os.path.exists(module_name):
        state = torch.load(module_name, map_location=torch.device('cpu'))

        NN2.load_state_dict(state['model'])

    NN1 = Module(NN_SIZE)
    module_name = 'Cylinder_200_les_les'

    if os.path.exists(module_name):
        state = torch.load(module_name, map_location=torch.device('cpu'))

        NN1.load_state_dict(state['model'])

    t_star = numerical_data['t_data']  # T x 1, [0 -> 16]
    x_star = numerical_data['x_data']  # N x 1
    y_star = numerical_data['y_data']  # N x 1
    U_star = numerical_data['U_data'][:, :, None]  # N x T
    V_star = numerical_data['V_data'][:, :, None]  # N x T
    P_star = numerical_data['P_data'][:, :, None]  # N x T
    C_star = numerical_data['C_data'][:, :, None]  # N x T

    N, T = U_star.shape[0], U_star.shape[1]
    x, y = x_star[:, 0], y_star[:, 0]

    for idx in np.arange(2, 16, 1):  # idx in np.arange(int(T / 10)) * 10:
        # t = t_star[idx][0]
        t = idx
        t_x_y = np.zeros((N, 3))

        t_x_y[:, 0] = t
        t_x_y[:, 1] = x
        t_x_y[:, 2] = y

        x_y = np.concatenate((x_star, y_star), axis=1)
        c = C_star
        var_c = np.var(c)
        c = c[:, 25, 0]

        my_plot(x, y, c)

        std_c = noisy_rate * np.sqrt(var_c)
        c = np.random.normal(c, std_c)

        my_plot(x, y, c)

        out = NN1(torch.FloatTensor(t_x_y)).detach().numpy()
        c_NN1 = out[:, 0]

        my_plot(x, y, c_NN1)

        out = NN2(torch.FloatTensor(t_x_y)).detach().numpy()
        c_NN2 = out[:, 0]

        my_plot(x, y, c_NN2)

        break

