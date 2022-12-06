import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ic_data = np.load('data/cell_migration/data_ic.npy')
loc = ic_data[0, :, 1]

ic_label = np.load('data/cell_migration/label_ic.npy')
ic_label = np.ones_like(ic_label[0]) * 0.0006
for i, x in enumerate(loc):
    if 500 < x < 1500:
        ic_label[i] = 0
ic_label = np.array([ic_label])

dx = 50
dt = 12
K = ((530.39, 0.066, -46.42),
     (484.74, 0.065, -43.15),
     (636.68, 0.070, -45.48),
     (982.26, 0.078, -47.65))


def time_step(c, k):
    k1, k2, k3 = k

    cd = np.insert(c, [0, -1], c[[0, -1]])
    c_xx = np.array([(cd[i - 1] + cd[i + 1] - 2 * cd[i]) / dx ** 2 for i in range(1, init.shape[0] + 1)])

    c_2 = c ** 2

    c_next = c + dt * (k1 * c_xx + k2 * c + k3 * c_2)

    return c_next


for init, k in zip(ic_label, K):
    c = init[:, 0]
    # c = savgol_filter(c, 21, 5)
    #
    # plt.plot(loc, c)
    # plt.show()

    c12 = time_step(c, k)
    c24 = time_step(c12, k)
    c36 = time_step(c24, k)
    c48 = time_step(c36, k)

    plt.plot(loc, c12)
    plt.plot(loc, c24)
    plt.plot(loc, c36)
    plt.plot(loc, c48)

    plt.show()
