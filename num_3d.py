from condition import t_x_y, u_v_p, dataset
from parameter import BATCH, device, collocation_size, train_size_rate, noisy_rate
import torch
import numpy as np
from torch.utils.data import DataLoader

rng_state = np.random.get_state()
np.random.shuffle(t_x_y)
np.random.set_state(rng_state)
np.random.shuffle(u_v_p)

var_cuvp = np.var(u_v_p, axis=0)
std_cuvp = noisy_rate * np.sqrt(var_cuvp)

norm_size = int(len(t_x_y) / 2)
rare_size = int(len(t_x_y) * train_size_rate)
# print(rare_size)

rare_data = t_x_y[:rare_size]
rare_label = u_v_p[:rare_size]
noisy_rare_label = np.random.normal(rare_label, std_cuvp)

validation_data = torch.FloatTensor(t_x_y[norm_size:norm_size + 10000]).requires_grad_(False).to(device)
validation_label = torch.FloatTensor(u_v_p[norm_size:norm_size + 10000]).requires_grad_(False).to(device)

# collocation_points = np.concatenate([16 * np.random.rand(collocation_size, 1),
#                                      10 * np.random.rand(collocation_size, 1) - 2.5,
#                                      5 * np.random.rand(collocation_size, 1) - 2.5], axis=1)
# collocation_points = torch.FloatTensor(collocation_points).requires_grad_(True).to(device)

rare_dataset = dataset(torch.FloatTensor(rare_data),
                       torch.FloatTensor(rare_label))
rare_dataloader_1 = DataLoader(dataset=rare_dataset,
                               batch_size=BATCH,
                               shuffle=True,
                               num_workers=8)
rare_dataloader_2 = DataLoader(dataset=rare_dataset,
                               batch_size=25 * BATCH,
                               shuffle=True,
                               num_workers=8)

noisy_rare_dataset = dataset(torch.FloatTensor(rare_data),
                             torch.FloatTensor(noisy_rare_label))
noisy_rare_dataloader_1 = DataLoader(dataset=noisy_rare_dataset,
                                     batch_size=BATCH,
                                     shuffle=True,
                                     num_workers=8)
noisy_rare_dataloader_2 = DataLoader(dataset=noisy_rare_dataset,
                                     batch_size=25 * BATCH,
                                     shuffle=True,
                                     num_workers=8)

if __name__ == '__main__':
    for x, y in rare_dataloader_2:
        print(x.shape, y.shape)