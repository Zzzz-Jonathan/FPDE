import os
import torch
from num_3d import loss_pde, loss_data, loss_icbc, x_star, y_star
from num_3d import rare_dataloader_2 as dataloader
from num_3d import validation_data, validation_label
from module import Module
from parameter import NN_SIZE, module_name, device, EPOCH, LOSS, sparse_num, ITERATION
from drawing.plt2img import fig2data
import numpy as np

from torch.utils.tensorboard import SummaryWriter

"""
This file is the main file used to train the conventional constrained NN for 2D cylindrical spoiler modeling. 
Furthermore, running this program is training the PINN using the given sparse data.
It is named ns_3d because the input to the network is three-dimensional (t, x, y), and the governed equations is N-S eq.
After preparing the data set, run the file directly to start training.

The relevant hyperparameters are quoted from parameter.py and need to be adjusted appropriately according to your own 
equipment conditions.

The network structure of cylindrical spoiler modeling is defined in cylinder/module.py.

PDE loss and other losses are defined in cylinder/num_3d.py. Sparse dataloader and data processing related comments are 
also in cylinder/num_3d.py.

The download method of training data is given in readme.md. Since the training data comes from 
other people’s open source data sets, users need to perform structured preprocessing first.

This experiment uses tensorboard to save the training process. Users can adjust the saving location in the 'path' 
variable according to their own needs.
"""

load = False
store = True
torch.manual_seed(3407)
# sparse_num = 'pure'
path = 'train_history/sparse/' + str(sparse_num) + '/ns'
module_name = path + '/' + module_name

if __name__ == '__main__':
    print(device)
    NN = Module(NN_SIZE).to(device)
    opt = torch.optim.SGD(params=NN.parameters(), lr=5e-2)

    start_epoch = 0

    writer = SummaryWriter(path)

    if os.path.exists(module_name + '_les_rec') and load:
        state = torch.load(module_name + '_les_rec')

        NN.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

        print('Start from epoch = %d' % start_epoch)

    min_loss = 1e6
    iter = 0
    x_grid, y_grid = x_star[:, 0], y_star[:, 0]
    t_x_y_grid = np.zeros((x_grid.shape[0], 3))
    t_x_y_grid[:, 0] = 8
    t_x_y_grid[:, 1] = x_grid
    t_x_y_grid[:, 2] = y_grid
    t_x_y_grid = torch.FloatTensor(t_x_y_grid).to(device)

    for epoch in range(start_epoch, EPOCH):
        for t_x_y, num_solution in dataloader:
            t_x_y = t_x_y.requires_grad_(True).to(device)
            num_solution = num_solution.requires_grad_(True).to(device)
            opt.zero_grad()
            iter += 1

            loss_u, loss_v, loss_div = loss_pde(NN, t_x_y)  # + loss_les(NN_les, t_x_y_col)
            pde_loss = loss_u + loss_v + loss_div  # + c_loss_u + c_loss_v + c_loss_div

            data_loss_1, data_loss_2 = loss_data(NN, t_x_y, num_solution)
            data_loss = data_loss_1 + data_loss_2

            ic_loss, bc_loss = loss_icbc(NN)
            icbc_loss = ic_loss + bc_loss

            loss = data_loss + pde_loss + icbc_loss

            validation_out = NN(validation_data)
            [va_u, va_v, va_p] = [LOSS(validation_out[:, 0], validation_label[:, 0]) / 3,
                                  LOSS(validation_out[:, 1], validation_label[:, 1]) / 3,
                                  LOSS(validation_out[:, 2], validation_label[:, 2]) / 3]

            validation_loss = va_u + va_v + va_p

            writer.add_scalars('1_loss', {'train': loss,
                                          'validation': validation_loss,
                                          'data_loss': data_loss_1,
                                          'std_loss': data_loss_2,
                                          'icbc_loss': icbc_loss}, iter)

            writer.add_scalars('pde_loss', {'loss': pde_loss,
                                            # 'loss_c': les_loss_c,
                                            'loss_u': loss_u,  # + c_loss_u,
                                            'loss_v': loss_v,  # + c_loss_v,
                                            'loss_div': loss_div,  # + c_loss_div
                                            }, iter)

            writer.add_scalars('validation_loss', {'total': validation_loss,
                                                   'u': va_u, 'v': va_v, 'p': va_p}, iter)

            test_out = NN(t_x_y_grid).cpu().detach().numpy()
            img = fig2data(x_grid, y_grid, test_out[:, 0], test_out[:, 1], test_out[:, 2])
            writer.add_image('%d' % iter, img, dataformats='HWC')

            loss.backward()  #
            opt.step()

            if store and iter % 50 == 0:
                state = {'model': NN.state_dict(),
                         'optimizer': opt.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name + '_rec')

            if validation_loss.item() < min_loss:
                min_loss = validation_loss.item()
                print('______Best loss model %.8f______' % loss.item())
                print('NS loss is %.8f' % pde_loss.item())
                print('DATA loss is %.8f' % data_loss_1.item())
                print('Validation loss is %.8f' % validation_loss.item())
                # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
                if store:
                    state = {'model': NN.state_dict(),
                             'optimizer': opt.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name)

            if iter > ITERATION:
                break

        print('------%d epoch------' % epoch)
        if iter > ITERATION:
            break
