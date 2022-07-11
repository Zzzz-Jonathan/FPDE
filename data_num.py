from condition import t_x_y, c_u_v_p, dataset
from parameter import BATCH, device
import torch
import numpy as np
from torch.utils.data import DataLoader


rng_state = np.random.get_state()
np.random.shuffle(t_x_y)
np.random.set_state(rng_state)
np.random.shuffle(c_u_v_p)

var_cuvp = np.var(c_u_v_p, axis=0)
std_cuvp = np.sqrt(var_cuvp)
print(len(std_cuvp))

norm_size = int(len(t_x_y) / 10)
rare_size = int(len(t_x_y) / 100)

rare_data = t_x_y[:rare_size]
rare_label = c_u_v_p[:rare_size]
noisy_rare_label = np.random.normal(rare_label, std_cuvp)

norm_data = t_x_y[:norm_size]
norm_label = c_u_v_p[:norm_size]
noisy_norm_label = np.random.normal(norm_label, std_cuvp)

rare_validation_data = torch.FloatTensor(t_x_y[rare_size:rare_size + 10000])
rare_validation_label = torch.FloatTensor(c_u_v_p[rare_size:rare_size + 10000])

rare_dataset = dataset(torch.FloatTensor(rare_data).requires_grad_(True).to(device), torch.FloatTensor(rare_label).to(device))
rare_dataloader = DataLoader(dataset=rare_dataset,
                             batch_size=BATCH,
                             shuffle=True,
                             num_workers=0)

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
    n = np.random.normal(a, [0.001, 0.1, 10, 100])
    print(n)
