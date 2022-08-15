import os
import plotly.graph_objects as go
import torch
from module import Module, ResLinear
from parameter import NN_SIZE_3D, module_name, noisy_3d_rate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import Rbf, griddata, RegularGridInterpolator
from stl import mesh
from mayavi import mlab
from mayavi.core.api import Engine

TIME = [0, 5, 10, 15]

normalize = False
c_norm = Normalize(vmin=0, vmax=0.8)
u_norm = Normalize(vmin=-0.2, vmax=1.4)
v_norm = Normalize(vmin=-0.8, vmax=0.8)
p_norm = Normalize(vmin=-0.6, vmax=0.6)
norm = None


def my_plot(_x, _y, _z, _n, name=None):
    X, Y, Z = np.mgrid[-3.5:5:170j, 0:5:100j, -2.5:2.5:100j]
    _X, _Y, _Z = np.linspace(-3.5, 5, 170), np.linspace(0, 5, 100), np.linspace(-2.5, 2.5, 100)
    _X2, _Y2, _Z2 = np.mgrid[-3.5:5:680j, 0:5:400j, -2.5:2.5:400j]

    values = griddata((_x, _y, _z), _n, (X, Y, Z), method='nearest')
    print('start plot')
    # lin = RegularGridInterpolator((_X, _Y, _Z), values, method='linear')
    # values = lin((_X2, _Y2, _Z2))

    # src = mlab.pipeline.scalar_field(values)
    # mlab.pipeline.iso_surface(src, contours=10, opacity=0.6,
    #                           extent=[-3.5, 5, 0, 5, -2.5, 2.5])
    #
    # mlab.show()

    fig = go.Figure(data=[go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        opacity=0.6,
        # isomax=15,
        # isomin=0,
        surface=dict(count=10),
        colorscale='hot',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ), mesh3D])

    fig.show()

    if name is not None:
        mlab.savefig('image/' + name + '_3d.png')


def stl2mesh3d(stl_mesh):
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)

    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3 * k for k in range(p)])
    J = np.take(ixr, [3 * k + 1 for k in range(p)])
    K = np.take(ixr, [3 * k + 2 for k in range(p)])

    return vertices, I, J, K


def scat(_x, _y, _z):
    plt.scatter(_x, _y, c=_z)
    plt.colorbar()

    plt.show()


my_mesh = mesh.Mesh.from_file('data/cylinder.stl')
vertices, I, J, K = stl2mesh3d(my_mesh)
x, y, z = vertices.T

colorscale = [[0, '#ffffff'], [1, '#ffffff']]
mesh3D = go.Mesh3d(
    x=x, y=y, z=z, i=I, j=J, k=K,
    flatshading=True,
    colorscale=colorscale,
    intensity=z,
    name='cylinder',
    showscale=False)

if __name__ == '__main__':
    NN2 = ResLinear(shape=[4, 20, 4])
    module_name_2 = 'train_history/4d/ns/' + 'Cylinder'

    NN1 = ResLinear(shape=[4, 20, 4])
    module_name_1 = 'train_history/4d/les/' + 'Cylinder'

    # NN1 = torch.nn.DataParallel(NN1)
    # NN2 = torch.nn.DataParallel(NN2)

    if os.path.exists(module_name_2):
        state = torch.load(module_name_2, map_location=torch.device('cpu'))

        NN2.load_state_dict(state['model'])
        print('load success')

    if os.path.exists(module_name_1):
        state = torch.load(module_name_1, map_location=torch.device('cpu'))

        NN1.load_state_dict(state['model'])
        print('load success')

    ref = np.load('data/re_expor/field_4096.npz')
    u, v, w, p = ref['u'], ref['v'], ref['w'], ref['p']
    N, T = u.shape[1], u.shape[0]

    d = np.load('data/re_expor/site.npz')
    x, y, z = d['x'][:, 0], d['y'][:, 0], d['z'][:, 0]
    re = np.log2(4096)

    t_x_y_z_re = np.zeros((N, 4))
    t_x_y_z_re[:, 1] = x
    t_x_y_z_re[:, 2] = y
    t_x_y_z_re[:, 3] = z
    # t_x_y_z_re[:, 4] = re

    # my_plot(x[:, 0], y[:, 0], z[:, 0], c[50, :, 0])

    for idx in np.arange(1, 2, 0.1):  # idx in np.arange(int(T / 10)) * 10:
        # t = t_star[idx][0]
        t = idx
        t_x_y = np.copy(t_x_y_z_re)
        t_x_y[:, 0] = t

        iidx = 0
        # name = ['u', 'v', 'w', 'p'][iidx]
        # c = [u, v, w, p][iidx]
        # c = c[int(idx / 0.02) - 1][:, 0]
        # var_c = np.var(c)
        #
        # my_plot(x, y, z, c)
        #
        # std_c = noisy_3d_rate * np.sqrt(var_c)
        # c = np.random.normal(c, std_c)
        #
        # my_plot(x, y, z, c)

        out = NN1(torch.FloatTensor(t_x_y)).detach().numpy()
        c_NN1 = out[:, iidx]

        my_plot(x, y, z, c_NN1)

        out = NN2(torch.FloatTensor(t_x_y)).detach().numpy()
        c_NN2 = out[:, iidx]

        my_plot(x, y, z, c_NN2)

        break
