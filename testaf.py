import os
from parameter import device
import numpy as np
import matplotlib.pyplot as plt
from num_af import dataset, data_mean, data_std, label_mean, label_std
from module import AF_module as Module
import torch

path = 'train_history/af/0/af/arterial_flow_rec'

(data, label, test_data, test_label, _, interface) = dataset
NN = Module()
if os.path.exists(path):
    state = torch.load(path, map_location=torch.device('cpu'))

    NN.load_state_dict(state['model'])

    print("Load success !")

test_data = test_data.view(1, -1, 2)
test_label = test_label.view(1, -1, 2)

# for i, x, y in zip([1, 2, 3, 3], data, label):
#     y_out = NN(x, i).detach().numpy()
#     y_ref = y.detach().numpy()
#     x = x.detach().numpy()[:, 0]
#
#     plt.figure(figsize=(3, 3))
#
#     # plt.plot(x, y_out[:, 0], color='red', ls='dashdot')
#     plt.plot(x, y_ref[:, 0], color='red', label='Area')
#     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#
#     plt.legend()
#     plt.show()
#     # plt.plot(x, y_out[:, 1], color='blue', ls='dashdot')
#
#     plt.figure(figsize=(3, 3))
#
#     plt.plot(x, y_ref[:, 1], color='blue', label='Velocity')
#     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#
#     plt.legend()
#     plt.show()

    # plt.plot(x, y_out[:, 2], color='green', ls='dashdot')
    # plt.show()

for i, (x, y) in enumerate(zip(test_data, test_label)):
    y_out = NN(x, 3) * label_std + label_mean
    y_out = y_out.detach().numpy()
    y_ref = y.detach().numpy()
    x = (x * data_std + data_mean).detach().numpy()[:, 0]

    plt.figure(figsize=(3, 3))

    plt.plot(x, y_out[:, 0], color='red', ls='dashdot', label='NN')
    plt.plot(x, y_ref[:, 0], color='red', label='measurement')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))

    plt.plot(x, y_out[:, 1], color='blue', ls='dashdot', label='NN')
    plt.plot(x, y_ref[:, 1], color='blue', label='measurement')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.legend()
    plt.show()

    plt.figure(figsize=(3, 3))

    plt.plot(x, y_out[:, 2], color='green', ls='dashdot', label='NN')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.legend()
    plt.show()

# for i in [1, 2, 3]:
#     y_out = NN(interface[0], i).detach().numpy()
#     x = interface[0].detach().numpy()[:, 0]
#
#     plt.plot(x, y_out[:, 0] * y_out[:, 1], label=('area%d' % i))
#
# plt.legend()
# plt.show()
