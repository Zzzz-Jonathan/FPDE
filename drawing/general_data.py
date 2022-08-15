import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

numerical_data = scipy.io.loadmat('../data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat')
x = numerical_data['x_data'][:, 0]  # N x 1
y = numerical_data['y_data'][:, 0]  # N x 1
U_star = numerical_data['U_data']  # N x T
V_star = numerical_data['V_data']  # N x T
P_star = numerical_data['P_data']  # N x T
C_star = numerical_data['C_data']  # N x T


def general_plot(idx):
    c = [C_star, U_star, V_star, P_star][idx + 1]
    c = c[:, 100]
    # c = P_star[:, 100] * V_star[:, 100]

    dxs = np.linspace(-2.5, 7.5, 10)
    dys = np.linspace(-2.5, 2.5, 5)
    dxs, dys = np.meshgrid(dxs, dys)
    z_new = griddata((x, y), c, (dxs, dys), method='nearest')

    z_filted = np.zeros_like(z_new)

    for i in range(z_new.shape[0]):
        for j in range(z_new.shape[1]):
            temp = [z_new[i, j]]
            if i - 1 >= 0:
                temp.append(z_new[i - 1, j])
                if j - 1 >= 0:
                    temp.append(z_new[i - 1, j - 1])
                if j + 1 < z_new.shape[1]:
                    temp.append(z_new[i - 1, j + 1])
            if j - 1 >= 0:
                temp.append(z_new[i, j - 1])
                if i + 1 < z_new.shape[0]:
                    temp.append(z_new[i + 1, j - 1])
            if i + 1 < z_new.shape[0]:
                temp.append(z_new[i + 1, j])
                if j + 1 < z_new.shape[1]:
                    temp.append(z_new[i + 1, j + 1])
            if j + 1 < z_new.shape[1]:
                temp.append(z_new[i, j + 1])

            z_filted[i, j] = np.mean(temp)

    norm = Normalize(vmin=-0.42, vmax=0.5)

    plt.imshow(z_filted, cmap=plt.get_cmap('seismic'), norm=norm)

    cylinder = plt.Circle(xy=(2, 2), radius=0.5, alpha=1, color='white')
    plt.gca().add_patch(cylinder)
    plt.colorbar()

    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == '__main__':
    # for i in [0, 1, 2]:
    general_plot(2)
