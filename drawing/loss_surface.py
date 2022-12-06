import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator


def fx(x, y):
    z = x ** 4 + y ** 4
    # z = x * (np.sin(4.5 * np.pi * x) + 1.5) + y * (np.sin(4.5 * np.pi * y) + 1.5)
    # z = (x + y)**0.5 * (1.5 * np.log2(y + 0.6)**2 * np.sin(3 * np.pi * x) + 1) + x * (y + 1) * (np.cos(3 * np.pi * y) + 1)
    return z


def fy(x, y):
    z = (1 - ((1-y) * x - 1)**4) ** 2 + (1 - ((1-y) * x - 1)**4) ** 2
    # z = x * (np.sin(4.5 * np.pi * x) + 1.5) + y * (np.sin(4.5 * np.pi * y) + 1.5)
    # z = x**0.5 * (1.5 * np.log2(x + 0.8)**2 * np.sin(4 * np.pi * y) + 1) + y * (x + 1) * (np.cos(4 * np.pi * x) + 1)
    return z


def plt3d(x, y, z, zlabel):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, z, cmap=plt.cm.get_cmap('turbo'))
    ax.invert_yaxis()
    # 前3个参数用来调整各坐标轴的缩放比例
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.2, 1]))
    # 分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
    ax.view_init(42, -32)

    ax.set_xlabel('$\Theta_1$')
    ax.set_ylabel('$\Theta_2$')
    ax.set_zlabel(zlabel)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(2))

    plt.show()


if __name__ == '__main__':
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')

    epsilon = 1e-5
    x = np.arange(0, 1 + epsilon, 0.001)
    y = np.arange(0, 1 + epsilon, 0.001)
    x, y = np.meshgrid(x, y)
    z1 = fx(x, y)
    z2 = fy(x, y)
    z = z1 + z2

    ax.plot_surface(x, y, z, cmap=plt.cm.get_cmap('turbo'))
    ax.invert_yaxis()
    # 前3个参数用来调整各坐标轴的缩放比例
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.2, 1]))
    # 分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
    ax.view_init(42, -32)

    ax.set_xlabel('$\Theta_1$')
    ax.set_ylabel('$\Theta_2$')
    ax.set_zlabel('$loss=loss_1+loss_2$')

    plt.show()

    plt3d(x, y, z1, '$loss_1$')
    plt3d(x, y, z2, '$loss_2$')

    # fig = plt.figure(figsize=(4, 3))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    #
    # ax.plot(y, fy(y), color='blue', label='$loss_1$')
    # ax.plot(x, fx(x), color='red', ls='-.', label='$loss_2$')
    # plt.legend()
    # plt.show()

    pass
