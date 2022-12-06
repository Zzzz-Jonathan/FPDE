# 6016935
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

numerical_data = scipy.io.loadmat('../data/Cylinder2D_Re200Pec2000_Neumann_Streaks.mat')
t_star = numerical_data['t_data']  # T x 1, [0 -> 16]
x_star = numerical_data['x_data']  # N x 1
y_star = numerical_data['y_data']  # N x 1

N, T = x_star.shape[0], t_star.shape[0]

t_x_y = np.zeros((N, T, 3))
for t in range(len(t_star)):
    t_x_y[:, t, 0] = t_star[t][0]

x_y = np.concatenate((x_star, y_star), axis=1)
for xy in range(len(x_y)):
    t_x_y[xy, :, 1] = x_y[xy][0]
    t_x_y[xy, :, 2] = x_y[xy][1]

t_x_y = t_x_y.reshape((N * T, 3))

np.random.shuffle(t_x_y)
t_x_y = t_x_y[:int(6016935 / 10)]  # t_x_y[:int(N * T * (2 ** -16))]

print(t_x_y.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection="3d")

xs = t_x_y[:, 1]
ys = t_x_y[:, 0]
zs = t_x_y[:, 2]

ax.scatter(xs, ys, zs, c='purple', marker=".", s=1)

ax.set(xlabel="X", ylabel="time", zlabel="Y")

ax.set_xlim([-2.5, 7.5])
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
ax.zaxis.label.set_size(16)

ax.yaxis.set_major_locator(MultipleLocator(4))
ax.set_ylim([0, 16])

ax.set_zlim([-2.5, 2.5])

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.9, 1.2, 0.8, 1]))
# 分别上下旋转和左右旋转，可以自己设置成一个比较好的参数
ax.view_init(32, -32)

plt.show()
