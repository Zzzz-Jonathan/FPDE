import numpy as np
import matplotlib.pyplot as plt
import torch
from module import Module
import os
from num_cm import data_mean, data_std, label_mean, label_std

test_data = [[[[t, x, n] for x in np.arange(15, 1875, 1)] for t in [0, 12, 24, 36, 48]] for n in [14, 16, 18, 20]]
test_data = np.array(test_data)
# test_data = test_data.reshape([-1, 3])

ref_data = np.load('data/cell_migration/data.npy')
ref_label = np.load('data/cell_migration/label.npy')

NN_SIZE = [3] + 5 * [100] + [1]
NN = Module(NN_SIZE)

path = 'train_history/cm/0.5/fcm/cell_migration'

if os.path.exists(path):
    state = torch.load(path, map_location=torch.device('cpu'))

    NN.load_state_dict(state['model'])

    print("Load success !")

for g in range(4):
    plt.figure(figsize=(10, 7))
    for pair, name, m in zip(range(5),
                             ['0', '12h', '24h', '36h', '48h'],
                             ['.', 's', '*', '^', 'x']):
        # if pair == 0:
        #     continue

        inp = torch.FloatTensor(test_data[g, pair])
        inp = (inp - data_mean) / data_std

        c_test = NN(inp)
        c_test = (c_test * label_std + label_mean).detach().numpy()

        plt.plot(test_data[g, pair, :, 1], c_test, label=name)
        plt.scatter(ref_data[g, pair, :, 1], ref_label[g, pair], s=50, label=name, marker=m)

        plt.tick_params(labelsize=30)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # plt.xlim(25, 1875)
    plt.legend()
    plt.show()
