import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.pyplot import MultipleLocator
from scipy.signal import savgol_filter
from pylab import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def my_colormap():
    clist = ['blue', 'red', 'blue']
    newcmp = LinearSegmentedColormap.from_list('my_bar', clist)

    return newcmp


class Smooth:
    def __init__(self, method=None, size=5):
        if method is None:
            self.method = self.average
        else:
            self.method = method

        self.size = size

    def average(self, x):
        # x = np.array(x)
        # axis = len(x.shape) - 1
        return np.average(x, axis=0)

    def smooth_xy(self, x):
        x = np.array(x)
        # new_x = np.zeros_like(x)
        new_x = savgol_filter(x, 211, 1)
        # kernel = []  # (size - 1) * [x[0]]
        # new_x = []
        #
        # for i, num in enumerate(x):
        #     if len(kernel) >= self.size:
        #         kernel.pop(0)
        #
        #     kernel.append(num)
        #     new_num = self.method(kernel)
        #     # print(new_num)
        #
        #     new_x.append(new_num)

        return np.array(new_x)

    def read_tensorboard(self, path):  # path为tensoboard文件的路径
        ea = event_accumulator.EventAccumulator(path)  # 初始化EventAccumulator对象
        ea.Reload()  # 将事件的内容都导进去
        return ea.scalars.Items(ea.scalars.Keys()[0])

    def read_file(self, path):
        for root, dirs, files in os.walk(path):
            return dirs

    def read_tbs(self, path):
        p1 = path + '/les'
        p2 = path + '/ns'

        names1 = self.read_file(p1)
        names2 = self.read_file(p2)
        print(names1)
        datas1 = {}
        datas2 = {}
        # print(names1, names2)
        for name in names1:
            log = self.read_tensorboard(p1 + '/' + name)
            value = np.array([i.value for i in log])
            smooth_value = self.smooth_xy(value)[:, None]
            step = np.array([i.step for i in log])[:, None]

            datas1[name] = np.concatenate((step, smooth_value, value[:, None]), axis=1)

        for name in names2:
            log = self.read_tensorboard(p2 + '/' + name)
            value = np.array([i.value for i in log])
            smooth_value = self.smooth_xy(value)[:, None]
            step = np.array([i.step for i in log])[:, None]

            datas2[name] = np.concatenate((step, smooth_value, value[:, None]), axis=1)

        return datas1, datas2

    def diff(self, x, y):
        K = []
        sign_K = []
        for i in range(len(x) - 1):
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            K.append(k)
            sign_K.append(0 if k <= 0 else 1)

        return np.array(K), np.array(sign_K)

    def loss_diff(self, path, names, clip):
        les_data, ns_data = self.read_tbs(path)
        K1 = []
        sign_K1 = []
        K2 = []
        sign_K2 = []
        for name in names:
            data1, data2 = les_data[name], ns_data[name]

            x1, y1 = data1[:clip[0], 0], data1[:clip[0], 2]
            k1, sign_k1 = self.diff(x1, y1)

            x2, y2 = data2[:clip[1], 0], data2[:clip[1], 2]
            k2, sign_k2 = self.diff(x2, y2)

            K1.append(k1)
            K2.append(k2)
            sign_K1.append(sign_k1)
            sign_K2.append(sign_k2)

        ans = []

        for i in range(int(len(K1) / 2)):
            k1, k2 = K1[2 * i], K1[2 * i + 1]
            sk1, sk2 = sign_K1[2 * i], sign_K1[2 * i + 1]
            xor = np.logical_xor(sk1, sk2)
            print('conflict rate: %.8f' % np.mean(xor))
            ans.append(np.mean(xor))
            conflict_cost = []
            for idx, x in enumerate(xor):
                if x == 1:
                    conflict_cost.append(max(k1[idx], k2[idx]))
            # print(conflict_cost)
            print('conflict cost: %.8f' % np.mean(conflict_cost))
            ans.append(np.mean(conflict_cost))

        for i in range(int(len(K2) / 2)):
            k1, k2 = K2[2 * i], K2[2 * i + 1]
            sk1, sk2 = sign_K2[2 * i], sign_K2[2 * i + 1]
            xor = np.logical_xor(sk1, sk2)
            print('conflict rate: %.8f' % np.mean(xor))
            ans.append(np.mean(xor))
            conflict_cost = []
            for idx, x in enumerate(xor):
                if x == 1:
                    conflict_cost.append(max(k1[idx], k2[idx]))
            # print(conflict_cost)
            print('conflict cost: %.8f' % np.mean(conflict_cost))
            ans.append(np.mean(conflict_cost))

        return ans


def my_kernel_function(x):
    return np.min(x, axis=0)


def im_bar(x, width=None):
    if width is None:
        width = int(x.shape[0] / 10)
    im_sign = np.zeros([x.shape[0], width])
    for i, x_ in enumerate(x):
        im_sign[i, :] = x_

    my_bar = my_colormap()

    plt.imshow(im_sign, cmap='seismic')  # my_bar)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    sm = Smooth(size=1)

    # name = [  # '../train_history/sparse/8',  # '../train_history/sparse/12', '../train_history/sparse/14',
    #     # '../train_history/sparse/15',  # '../train_history/sparse/16',
    #     '../train_history/sparse/10',
    #     '../train_history/noisy/1', '../train_history/noisy/2', '../train_history/noisy/3',
    #     '../train_history/noisy/4', '../train_history/noisy/5', '../train_history/noisy/6']
    # clip = [  # [15000, 25000],  # [25000, 20000], [15000, 15000],
    #     # [7500, 30000],
    #     [50000, 50000],
    #     [10000, 10000], [10000, 10000], [10000, 10000],
    #     [10000, 10000], [10000, 10000], [10000, 10000]]
    #
    # Cans = []
    # for n, cp in zip(name, clip):
    #     Cans.append(sm.loss_diff(n, ['1_loss_data_loss', 'pde_loss_loss'], cp))
    # Cans = np.array(Cans)
    #
    # x = np.arange(len(name))
    # width = 0.3
    # x1 = x - width / 2
    # x2 = x + width / 2
    #
    # fig = plt.figure(figsize=(6, 3))
    # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax1.bar(x1, Cans[:, 0], width=width, label='FPDE', color='red')
    # ax1.bar(x2, Cans[:, 2], width=width, label='PDE', color='blue')
    # # ax1.set_xlabel('Data Points Num')
    # ax1.set_ylabel('Conflict rate')
    # ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    # # plt.xticks(x, [  # 'sparse $2^{-8}$',  # 'sparse $2^{-12}$', 'sparse $2^{-14}$',
    # #     # 'sparse $2^{-15}$',  # 'sparse $2^{-16}$',
    # #     'noisy 25%', 'noisy 50%', 'noisy 75%', 'noisy 100%', 'noisy 125%', 'noisy 150%'
    # # ])
    # plt.xticks([])
    # ax1.set_xlabel('data accuracy (0 -> 150% noisy rate)')
    # plt.legend()
    # plt.show()
    #
    # fig = plt.figure(figsize=(6, 3))
    # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax1.bar(x1, Cans[:, 1], width=width, label='FPDE', color='red')
    # ax1.bar(x2, Cans[:, 3], width=width, label='PDE', color='blue')
    # # ax1.set_xlabel('Data Points Num')
    # ax1.set_ylabel('Conflict mean cost')
    # ax1.yaxis.set_major_locator(MultipleLocator(0.001))
    # # plt.xticks(x, [  # 'sparse $2^{-8}$',  # 'sparse $2^{-12}$', 'sparse $2^{-14}$',
    # #     # 'sparse $2^{-15}$',  # 'sparse $2^{-16}$',
    # #     'noisy 25%', 'noisy 50%', 'noisy 75%', 'noisy 100%', 'noisy 125%', 'noisy 150%'
    # # ])
    # plt.xticks([])
    # ax1.set_xlabel('data accuracy (0 -> 150% noisy rate)')
    # plt.legend()
    # plt.show()

    # d1, d2 = sm.read_tbs('../train_history/noisy/6')
    # a, b = d1['1_loss_data_loss'], d1['pde_loss_loss']
    # c, d = d2['1_loss_data_loss'], d2['pde_loss_loss']
    #
    # np.savez('sparse_13_loss.npz', data_1=a, pde_1=b, data_2=c, pde_2=d)

    dd = np.load('sparse_13_loss.npz')
    data_1, pde_1 = dd['data_1'], dd['pde_1']
    data_2, pde_2 = dd['data_2'], dd['pde_2']

    _, sign_data_1 = sm.diff(data_1[:, 0], data_1[:, 2])
    _, sign_pde_1 = sm.diff(pde_1[:, 0], pde_1[:, 2])
    _, sign_data_2 = sm.diff(data_2[:, 0], data_2[:, 2])
    _, sign_pde_2 = sm.diff(pde_2[:, 0], pde_2[:, 2])

    lim_1, lim_2 = 0, 0
    for i, x in enumerate(data_1[:, 0]):
        if x >= 15000:
            lim_1 = i
            break

    for i, x in enumerate(data_2[:, 0]):
        if x >= 15000:
            lim_2 = i
            break

    # im_bar(sign_data_1[:lim_1])
    # im_bar(sign_data_2[:lim_2])
    # im_bar(sign_pde_1[:lim_1])
    # im_bar(sign_pde_2[:lim_2])
    im_bar(np.logical_xor(sign_data_1[:lim_1], sign_pde_1[:lim_1]))
    im_bar(np.logical_xor(sign_data_2[:lim_2], sign_pde_2[:lim_2]))

    # fig = plt.figure(figsize=(4, 3))
    # ax1 = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    # ax1.plot(data_1[:, 0], data_1[:, 1], label='FPDE data loss', color='red', alpha=0.7)
    # # ax1.plot(data_2[:, 0], data_2[:, 1], label='PDE data loss', color='red', alpha=0.7)
    # ax1.set_xlabel('train')
    # ax1.set_ylabel('Data loss')
    # ax1.set_ylim([0, 0.1])
    # ax1.set_xlim([0, 2500])
    # ax1.yaxis.set_major_locator(MultipleLocator(0.02))
    # # plt.xticks(x, x_label)
    # # plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    #
    # ax2 = plt.twinx()
    # ax2.set_ylim([0, 0.02])
    # ax2.set_xlim([0, 2500])
    # ax2.plot(pde_1[:, 0], pde_1[:, 1], label='FPDE pde loss', ls='-.', color='blue', alpha=0.7)
    # # ax2.plot(pde_2[:, 0], pde_2[:, 1], label='PDE pde loss', ls='-.', color='blue', alpha=0.7)
    # ax2.set_ylabel('PDE loss')
    # ax2.yaxis.set_major_locator(MultipleLocator(0.005))
    #
    # fig.legend(bbox_to_anchor=(0.8, 0.8))
    plt.show()
    pass

    # print(1)
