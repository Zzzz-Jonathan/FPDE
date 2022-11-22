import torch
import numpy as np
from torch.utils.data import DataLoader
from parameter import size_3d_rate, LOSS, device, noisy_3d_rate, BATCH_3d, dataset, gradients


def gauss_kernel(size=5, sigma=1):
    X = np.linspace(-3 * sigma, 3 * sigma, size)
    Y = np.linspace(-3 * sigma, 3 * sigma, size)
    Z = np.linspace(-3 * sigma, 3 * sigma, size)
    x, y, z = np.meshgrid(X, Y, Z)

    kernel = 1 / (2 * np.pi * sigma ** 2) ** (3 / 2) * np.exp(- (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    Z = kernel.sum()  # 计算归一化系数
    kernel = (1 / Z) * kernel

    return kernel


def loss_pde(nn, data_inp):
    [t, x, y, z, re] = torch.split(data_inp, 1, dim=1)
    data_inp = torch.cat([t, x, y, z, re], dim=1)
    re = 2 ** re

    out = nn(data_inp.to(device))
    out = out * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]

    [u, v, w, p] = torch.split(out, 1, dim=1)
    zeros = torch.zeros_like(u)

    u_t = gradients(u, t)
    v_t = gradients(v, t)
    w_t = gradients(w, t)

    u_x = gradients(u, x)
    v_x = gradients(v, x)
    w_x = gradients(w, x)
    p_x = gradients(p, x)

    u_y = gradients(u, y)
    v_y = gradients(v, y)
    w_y = gradients(w, y)
    p_y = gradients(p, y)

    u_z = gradients(u, z)
    v_z = gradients(v, z)
    w_z = gradients(w, z)
    p_z = gradients(p, z)

    u_xx = gradients(u_x, x)
    v_xx = gradients(v_x, x)
    w_xx = gradients(w_x, x)

    u_yy = gradients(u_y, y)
    v_yy = gradients(v_y, y)
    w_yy = gradients(w_y, y)

    u_zz = gradients(u_z, z)
    v_zz = gradients(v_z, z)
    w_zz = gradients(w_z, z)

    l1 = LOSS(u_t + (u * u_x + v * u_y + w * u_z) + p_x - (10 / re) * (u_xx + u_yy + u_zz), zeros)
    l2 = LOSS(v_t + (u * v_x + v * v_y + w * v_z) + p_y - (10 / re) * (v_xx + v_yy + v_zz), zeros)
    l3 = LOSS(w_t + (u * w_x + v * w_y + w * w_z) + p_z - (10 / re) * (w_xx + w_yy + w_zz), zeros)
    l4 = LOSS(u_x + v_y + w_z, zeros)

    return l1, l2, l3, l4


def loss_icbc(nn, size=3000):
    np.random.shuffle(ic_v)
    ic_v_t = torch.FloatTensor(ic_v[:size]).to(device)
    ic_v_out = nn(ic_v_t) * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    ic_v_out = ic_v_out[:, :3]
    ic_v_l = torch.FloatTensor([10, 0, 0]).to(device) * torch.ones_like(ic_v_out).to(device)

    np.random.shuffle(bc_c)
    bc_c_t = torch.FloatTensor(bc_c[:size]).to(device)
    bc_c_out = nn(bc_c_t) * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    bc_c_out = bc_c_out[:, :3]
    bc_c_l = torch.zeros_like(bc_c_out).to(device)

    bc_top_t, bc_btm_t = my_shuffle(bc_top, bc_btm, size)
    out_top = nn(torch.FloatTensor(bc_top_t).to(device))
    out_top = out_top * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    # out_top = out_top[:, :3]

    out_btm = nn(torch.FloatTensor(bc_btm_t).to(device))
    out_btm = out_btm * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    # out_btm = out_btm[:, :3]

    bc_fot_t, bc_bak_t = my_shuffle(bc_fot, bc_bak, size)
    out_fot = nn(torch.FloatTensor(bc_fot_t).to(device))
    out_fot = out_fot * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    # out_fot = out_fot[:, :3]

    out_bak = nn(torch.FloatTensor(bc_bak_t).to(device))
    out_bak = out_bak * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    # out_bak = out_bak[:, :3]

    return LOSS(ic_v_l, ic_v_out) + LOSS(bc_c_l, bc_c_out) + LOSS(out_top, out_btm) + LOSS(out_bak, out_fot)


def loss_data(nn, data_inp, label):
    out = nn(data_inp.to(device))

    out = out * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
    label = label * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]

    return LOSS(out, label), LOSS(out.std(axis=0), label.std(axis=0))


def loss_les(nn, data_inp, dx=0.05, dy=0.05, dz=0.05):
    [t, x, y, z, re] = torch.split(data_inp, 1, dim=1)

    re = 2 ** re

    xxs = [x, x + dx, x - dx]
    yys = [y, y + dy, y - dy]
    zzs = [z, z + dz, z - dz]

    txys = []

    for xs in xxs:
        for ys in yys:
            for zs in zzs:
                txys.append(torch.cat([t, xs, ys, zs, re], dim=1))

    nn_outs = []
    nn_uus = []
    nn_uvs = []
    nn_uws = []
    nn_vvs = []
    nn_vws = []
    nn_wws = []

    for txy, weight in zip(txys, weights):
        out = nn(txy.to(device))
        out = out * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]

        w = kernel[center + weight[0]][center + weight[1]][center + weight[2]]
        nn_outs.append(w * out)
        nn_uus.append(w * out[:, 0] * out[:, 0])
        nn_uvs.append(w * out[:, 0] * out[:, 1])
        nn_uws.append(w * out[:, 0] * out[:, 2])

        nn_vvs.append(w * out[:, 1] * out[:, 1])
        nn_vws.append(w * out[:, 1] * out[:, 2])

        nn_wws.append(w * out[:, 2] * out[:, 2])

    out_bar = sum(nn_outs)
    uu_bar = sum(nn_uus)
    uv_bar = sum(nn_uvs)
    uw_bar = sum(nn_uws)

    vv_bar = sum(nn_vvs)
    vw_bar = sum(nn_vws)

    ww_bar = sum(nn_wws)

    [u_bar, v_bar, w_bar, p_bar] = torch.split(out_bar, 1, dim=1)

    u_bar_t = gradients(u_bar, t)
    v_bar_t = gradients(v_bar, t)
    w_bar_t = gradients(w_bar, t)

    uu_bar_x = gradients(uu_bar, x)
    uv_bar_x = gradients(uv_bar, x)
    uw_bar_x = gradients(uw_bar, x)

    uv_bar_y = gradients(uv_bar, y)
    vv_bar_y = gradients(vv_bar, y)
    vw_bar_y = gradients(vw_bar, y)

    uw_bar_z = gradients(uw_bar, z)
    vw_bar_z = gradients(vw_bar, z)
    ww_bar_z = gradients(ww_bar, z)

    p_bar_x = gradients(p_bar, x)
    p_bar_y = gradients(p_bar, y)
    p_bar_z = gradients(p_bar, z)

    u_bar_x = gradients(u_bar, x)
    v_bar_y = gradients(v_bar, y)
    w_bar_z = gradients(w_bar, z)

    u_bar_xx = gradients(u_bar_x, x)
    u_bar_yy = gradients(u_bar, y, order=2)
    u_bar_zz = gradients(u_bar, z, order=2)

    v_bar_xx = gradients(v_bar, x, order=2)
    v_bar_yy = gradients(v_bar_y, y)
    v_bar_zz = gradients(v_bar, z, order=2)

    w_bar_xx = gradients(w_bar, x, order=2)
    w_bar_yy = gradients(w_bar, y, order=2)
    w_bar_zz = gradients(w_bar_z, z)

    zeros = torch.zeros_like(u_bar)

    l1 = LOSS(u_bar_t + (uu_bar_x + uv_bar_y + uw_bar_z) + p_bar_x - (10 / re) * (u_bar_xx + u_bar_yy + u_bar_zz),
              zeros)
    l2 = LOSS(v_bar_t + (uv_bar_x + vv_bar_y + vw_bar_z) + p_bar_y - (10 / re) * (v_bar_xx + v_bar_yy + v_bar_zz),
              zeros)
    l3 = LOSS(w_bar_t + (uw_bar_x + vw_bar_y + ww_bar_z) + p_bar_z - (10 / re) * (w_bar_xx + w_bar_yy + w_bar_zz),
              zeros)
    l4 = LOSS(u_bar_x + v_bar_y + w_bar_z, zeros)

    return l1, l2, l3, l4


def load_label(path):
    data = np.load(path)

    u, v, w, p = data['u'], data['v'], data['w'], data['p']
    u_v_w_p = np.concatenate((u, v, w, p), axis=2)

    T, N = u_v_w_p.shape[0], u_v_w_p.shape[1]
    u_v_w_p = u_v_w_p.reshape((T * N, 4))

    return u_v_w_p


def load_data(path, re):
    data = np.load(path)

    t, x, y, z = data['t'], data['x'], data['y'], data['z']
    T, N = t.shape[0], x.shape[0]

    t_x_y_z_re = np.zeros((T, N, 5))

    for i, time in enumerate(t):
        t_x_y_z_re[i, :, 0] = time[0]

    x_y_z = np.concatenate((x, y, z), axis=1)
    for i, xyz in enumerate(x_y_z):
        t_x_y_z_re[:, i, 1] = xyz[0]
        t_x_y_z_re[:, i, 2] = xyz[1]
        t_x_y_z_re[:, i, 3] = xyz[2]

    t_x_y_z_re[:, :, 4] = re

    t_x_y_z_re = t_x_y_z_re.reshape((T * N, 5))

    return t_x_y_z_re


def shuffle_clip(path, re, idx):
    uvwp = load_label(path)
    txyz = load_data('data/re_expor/site.npz', re)

    rng_state = np.random.get_state()
    np.random.shuffle(txyz)
    np.random.set_state(rng_state)
    np.random.shuffle(uvwp)

    full_size = txyz.shape[0]
    train_size = int(full_size * size_3d_rate[idx])

    train_data = txyz[:train_size]
    train_label = uvwp[:train_size]

    validation_data = txyz[train_size:train_size + 20000]
    validation_label = uvwp[train_size:train_size + 20000]

    return train_data, train_label, validation_data, validation_label


def load():
    re_nums = range(2, 15)
    train_label, train_data, validation_label, validation_data = [], [], [], []
    for i, num in enumerate(re_nums):
        path = 'data/re_expor/field_' + str(2 ** num) + '.npz'
        [t_d, t_l, v_d, v_l] = shuffle_clip(path, num, i)
        print(t_d.shape, v_l.shape)
        train_label.append(t_l)
        train_data.append(t_d)
        validation_label.append(v_l)
        validation_data.append(v_d)

    train_label = np.concatenate(train_label, axis=0)
    train_data = np.concatenate(train_data, axis=0)
    validation_label = np.concatenate(validation_label, axis=0)
    validation_data = np.concatenate(validation_data, axis=0)

    print(train_label.shape, validation_data.shape)

    np.save('data/re_expor/5d/training_data.npy', train_data)
    np.save('data/re_expor/5d/training_label.npy', train_label)
    np.save('data/re_expor/5d/validation_data.npy', validation_data)
    np.save('data/re_expor/5d/validation_label.npy', validation_label)


def noisy_save():
    # td = np.load('data/re_expor/training_data.npy')
    tl = np.load('data/re_expor/5d/training_norm_label.npy')
    # vd = np.load('data/re_expor/validation_data.npy')
    # vl = np.load('data/re_expor/validation_label.npy')

    var = np.var(tl, axis=0)
    tl = np.random.normal(tl, np.sqrt(var) * noisy_3d_rate)

    np.save('data/re_expor/5d/training_noisy_norm_label.npy', tl)


def norm_save():
    data = np.load('data/re_expor/5d/training_data.npy')
    label = np.load('data/re_expor/5d/training_label.npy')

    va_label = np.load('data/re_expor/5d/validation_label.npy')
    va_data = np.load('data/re_expor/5d/validation_data.npy')

    np_norm = np.concatenate((np.max(label, axis=0)[:, None], np.min(label, axis=0)[:, None]), axis=1)
    label = (label - np_norm[:, 1]) / (np_norm[:, 0] - np_norm[:, 1])
    va_label = (va_label - np_norm[:, 1]) / (np_norm[:, 0] - np_norm[:, 1])
    # print(np.max(va_label, axis=0), np.min(va_label, axis=0))

    np.save('data/re_expor/5d/training_norm_label.npy', label)
    np.save('data/re_expor/5d/validation_norm_label.npy', va_label)
    np.save('data/re_expor/5d/norm_para.npy', np_norm)


def icbc_data_save():
    x_min = -3.5
    y_min, y_max = 0, 5
    z_min, z_max = -2.5, 2.5

    ic_data = np.array([[t, x_min, y, z, re] for t in np.arange(0, 2.3, 0.2)
                        for y in np.arange(-0.5, 5.6, 0.5)
                        for z in np.arange(-3, 3.1, 0.5)
                        for re in np.arange(1, 15, 0.5)])

    bc_data_c = np.array([[t, (np.cos(i) / 2) - 1, y, np.sin(i) / 2, re] for t in np.arange(0, 2.3, 0.2)
                          for i in np.linspace(0, 2 * np.pi, 16)
                          for y in np.arange(-0.5, 5.6, 0.5)
                          for re in np.arange(1, 15, 0.5)])

    bc_data_top = np.array([[t, x, y_max, z, re] for t in np.arange(0, 2.3, 0.2)
                            for x in np.arange(-4, 5.6, 0.5)
                            for z in np.arange(-3, 3.1, 0.5)
                            for re in np.arange(1, 15, 0.5)])
    bc_data_btm = np.copy(bc_data_top)
    bc_data_btm[:, 2] = y_min

    bc_data_fot = np.array([[t, x, y, z_max, re] for t in np.arange(0, 2.3, 0.2)
                            for x in np.arange(-4, 5.6, 0.5)
                            for y in np.arange(-0.5, 5.6, 0.5)
                            for re in np.arange(1, 15, 0.5)])
    bc_data_bak = np.copy(bc_data_fot)
    bc_data_bak[:, 3] = z_min

    print(ic_data.shape, bc_data_c.shape, bc_data_bak.shape)

    np.savez('data/re_expor/5d/icbc.npz', ic_data=ic_data, bc_data_c=bc_data_c,
             bc_data_top=bc_data_top, bc_data_btm=bc_data_btm,
             bc_data_fot=bc_data_fot, bc_data_bak=bc_data_bak)


def my_shuffle(d, l, size):
    rng_state = np.random.get_state()
    np.random.shuffle(d)
    np.random.set_state(rng_state)
    np.random.shuffle(l)

    return d[:size], l[:size]


kernel = gauss_kernel(size=3)
center = int(kernel.shape[0] / 2)
xs_idx = [0, 1, 1] if center == 1 else [0, 1, 1, 2, 2]
ys_idx = [0, 1, 1] if center == 1 else [0, 1, 1, 2, 2]
zs_idx = [0, 1, 1] if center == 1 else [0, 1, 1, 2, 2]
weights = [[i, j, k] for i in xs_idx for j in ys_idx for k in zs_idx]

t_data = np.load('data/re_expor/5d/training_data.npy')
# [-3.5 -> 5] [0 -> 5] [-2.5 -> 2.5]
# [-1 0 5] r = 1
t_n_label = np.load('data/re_expor/5d/training_noisy_norm_label.npy')
# t_label = np.load('data/re_expor/5d/training_label.npy')
v_data = np.load('data/re_expor/5d/validation_data.npy')
v_label = np.load('data/re_expor/5d/validation_norm_label.npy')
norm_para = torch.FloatTensor(np.load('data/re_expor/5d/norm_para.npy')).to(device)

icbc_data = np.load('data/re_expor/5d/icbc.npz')
ic_v = icbc_data['ic_data']
bc_c = icbc_data['bc_data_c']
bc_top = icbc_data['bc_data_top']
bc_btm = icbc_data['bc_data_btm']
bc_fot = icbc_data['bc_data_fot']
bc_bak = icbc_data['bc_data_bak']

# validation_data = torch.FloatTensor(v_data).to(device)
# validation_label = torch.FloatTensor(v_label).to(device)

noisy_dataset = dataset(torch.FloatTensor(t_data),
                        torch.FloatTensor(t_n_label))
noisy_dataloader_1 = DataLoader(dataset=noisy_dataset,
                                batch_size=BATCH_3d,
                                shuffle=True,
                                num_workers=8)
noisy_dataloader_2 = DataLoader(dataset=noisy_dataset,
                                batch_size=25 * BATCH_3d,
                                shuffle=True,
                                num_workers=8)

# simple_dataset = dataset(torch.FloatTensor(t_data),
#                          torch.FloatTensor(t_label))
# simple_dataloader_1 = DataLoader(dataset=simple_dataset,
#                                  batch_size=BATCH_3d,
#                                  shuffle=True,
#                                  num_workers=8)
# simple_dataloader_2 = DataLoader(dataset=simple_dataset,
#                                  batch_size=25 * BATCH_3d,
#                                  shuffle=True,
#                                  num_workers=8)

if __name__ == '__main__':
    # icbc_data_save()
    # a = np.load('data/re_expor/training_data.npy')
    # b = np.load('data/re_expor/training_noisy_label.npy')
    # print(a.shape, b.shape)
    # c = np.load('data/re_expor/field_16384.npz')
    # c = c['u']
    #
    # d = np.load('data/re_expor/site.npz')
    # x, y, z = d['x'], d['y'], d['z']
    #
    # my_plot(x[:, 0], y[:, 0], z[:, 0], c[50, :, 0])

    pass
