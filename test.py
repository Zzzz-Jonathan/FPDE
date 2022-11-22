import os
import torch
from module import Module
from parameter import NN_SIZE, module_name, noisy_rate
from condition import numerical_data, lab
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import Rbf, griddata
import scipy.io

TIME = [0, 5, 10, 15]

normalize = True
c_norm = Normalize(vmin=0, vmax=0.8)
u_norm = Normalize(vmin=-0.2, vmax=1.4)
v_norm = Normalize(vmin=-0.8, vmax=0.8)
p_norm = Normalize(vmin=-0.6, vmax=0.6)
norm = None


def my_plot(_x, _y, _z, name=None):
    plt.figure(figsize=(12, 6))

    dxs = np.linspace(-2.5, 7.5, 1000)
    dys = np.linspace(-2.5, 2.5, 500)
    dxs, dys = np.meshgrid(dxs, dys)

    z_new = griddata((_x, _y), _z, (dxs, dys), method='linear')

    if name is not None and normalize:
        exec('global norm; norm = %s_norm' % name[-1])

    # plt.imshow(z_new, cmap=plt.get_cmap('seismic'), norm=None)
    # C = plt.contour(dxs, dys, z_new, 40, colors='black')
    plt.contour(dxs, dys, z_new, 30,
                cmap=plt.get_cmap('gist_ncar'),
                zorder=1, norm=norm, linewidths=3)
    # plt.clabel(C, inline=1, fontsize=8)
    # plt.colorbar()
    # plt.contourf(dxs, dys, z_new, levels=50, cmap=plt.get_cmap('Spectral'))

    cylinder = plt.Circle(xy=(0, 0), radius=0.5, alpha=1, color='black', zorder=2)
    plt.gca().add_patch(cylinder)

    if name is not None:
        plt.savefig('image/' + name + '.png')

    plt.show()


def scat(_x, _y, _z):
    plt.scatter(_x, _y, c=_z)
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    NN2 = Module(NN_SIZE)
    module_name = 'train_history/noisy/6/ns/Cylinder'

    if os.path.exists(module_name):
        state = torch.load(module_name, map_location=torch.device('cpu'))

        NN2.load_state_dict(state['model'])
        print('load success')

    NN1 = Module(NN_SIZE)
    module_name = 'train_history/noisy/6/les/Cylinder'

    if os.path.exists(module_name):
        state = torch.load(module_name, map_location=torch.device('cpu'))

        NN1.load_state_dict(state['model'])
        print('load success')

    t_star = numerical_data['t' + lab]  # T x 1, [0 -> 16]
    x_star = numerical_data['x' + lab]  # N x 1
    y_star = numerical_data['y' + lab]  # N x 1
    U_star = numerical_data['U' + lab][:, :, None]  # N x T
    V_star = numerical_data['V' + lab][:, :, None]  # N x T
    P_star = numerical_data['P' + lab][:, :, None]  # N x T
    # C_star = numerical_data['C' + lab][:, :, None]  # N x T

    N, T = U_star.shape[0], U_star.shape[1]
    x, y = x_star[:, 0], y_star[:, 0]

    for idx in np.arange(8, 16, 1):  # idx in np.arange(int(T / 10)) * 10:
        # t = t_star[idx][0]
        t = idx
        t_x_y = np.zeros((N, 3))

        t_x_y[:, 0] = t
        t_x_y[:, 1] = x
        t_x_y[:, 2] = y

        iidx = 2
        name = ['u', 'v', 'p'][iidx]
        c = [U_star, V_star, P_star][iidx]

        x_y = np.concatenate((x_star, y_star), axis=1)
        var_c = np.var(c)
        c = c[:, 100, 0]

        my_plot(x, y, c, name='ori_' + name)

        # std_c = noisy_rate * np.sqrt(var_c)
        # nc = np.random.normal(c, std_c)
        #
        # my_plot(x, y, nc, name='noi_' + name)

        # out = NN1(torch.FloatTensor(t_x_y)).detach().numpy()
        # c_NN1 = out[:, iidx]
        #
        # my_plot(x, y, c_NN1, name='les_' + name)
        #
        # out = NN2(torch.FloatTensor(t_x_y)).detach().numpy()
        # c_NN2 = out[:, iidx]
        #
        # my_plot(x, y, c_NN2, name='ns_' + name)

        break
