import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from module import Module
from parameter import NN_SIZE, module_name
from condition import numerical_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, griddata

idx = 10


class GaussianBlur(torch.nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        # kernel = [[0.03797616, 0.044863533, 0.03797616],
        #           [0.044863533, 0.053, 0.044863533],
        #           [0.03797616, 0.044863533, 0.03797616]]
        # kernel = [[1, 1, 1],
        #           [1, 1, 1],
        #           [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x.unsqueeze(1), self.weight, padding=2)


if __name__ == '__main__':
    pass
    # NN = Module(NN_SIZE)
    # # writer = SummaryWriter('/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history')
    #
    # if os.path.exists(module_name):
    #     state = torch.load(module_name)
    #
    #     NN.load_state_dict(state['model'])
    #
    # gb = GaussianBlur()
    #
    # t_star = numerical_data['t_data']  # T x 1, [0 -> 16]
    # x_star = numerical_data['x_data']  # N x 1
    # y_star = numerical_data['y_data']  # N x 1
    # U_star = numerical_data['U_data'][:, :, None]  # N x T
    # V_star = numerical_data['V_data'][:, :, None]  # N x T
    # P_star = numerical_data['P_data'][:, :, None]  # N x T
    # C_star = numerical_data['C_data'][:, :, None]  # N x T
    #
    # N, T = U_star.shape[0], U_star.shape[1]
    # x, y = x_star[:, 0], y_star[:, 0]
    #
    # c = C_star[:, idx, 0]
    #
    # t = t_star[idx][0]
    # # t = idx
    # t_x_y = np.zeros((N, 3))
    #
    # t_x_y[:, 0] = t
    # t_x_y[:, 1] = x
    # t_x_y[:, 2] = y
    #
    # out = NN(torch.FloatTensor(t_x_y)).detach().numpy()
    # c_NN = out[:, 0]
    #
    # dxs = np.linspace(-2.5, 7.5, 1000)
    # dys = np.linspace(-2.5, 2.5, 500)
    # dxs, dys = np.meshgrid(dxs, dys)
    #
    # z_new = griddata((x, y), c, (dxs, dys), method='linear')
    # plt.imshow(z_new, cmap=plt.get_cmap('hot'))
    # plt.colorbar()
    # plt.show()
    #
    # gb_z = torch.FloatTensor(z_new).unsqueeze(0)
    #
    # gb_z = gb(gb_z)
    # gb_z = gb_z[0, 0, :, :].numpy()
    # print(gb_z.shape)
    #
    # plt.imshow(gb_z, cmap=plt.get_cmap('hot'))
    # plt.colorbar()
    # plt.show()
    #
    # after_gb = z_new - gb_z
    # plt.imshow(after_gb, cmap=plt.get_cmap('hot'))
    # plt.colorbar()
    # plt.show()
