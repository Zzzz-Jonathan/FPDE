import os
import torch
from num_cm import loss_pde, loss_data, loss_neumann_bc, dataset, full_data, loss_fpde
from module import Module
from parameter import module_name, device, EPOCH, LOSS, sparse_cm, LR, ITERATION
import numpy as np

from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
Filter = True
path = 'train_history/cm/' + str(sparse_cm) + ('/fcm' if Filter else '/cm')
module_name = path + '/cell_migration'

if __name__ == '__main__':
    print(device)
    NN_SIZE = [3] + 5 * [100] + [1]
    NN = Module(NN_SIZE).to(device)
    opt = torch.optim.Adam(params=NN.parameters(), lr=LR)

    start_epoch = 0

    writer = SummaryWriter(path)

    (train_data, train_label, val_data, val_label, bc_data) = dataset

    if os.path.exists(module_name + 'cm_rec') and load:
        state = torch.load(module_name + 'cm_rec')

        NN.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

        print('Start from epoch = %d' % start_epoch)

    min_loss = 1e6
    iter = 0

    # tensorboard --logdir=./train_history/cm/2/cm --port 14514

    for epoch in range(start_epoch, EPOCH):
        opt.zero_grad()
        iter += 1

        pde_loss = None
        if Filter is True:
            pde_loss = loss_fpde(NN, full_data)
        else:
            pde_loss = loss_pde(NN, full_data)

        data_loss = loss_data(NN, train_data, train_label)

        bc_loss = loss_neumann_bc(NN, bc_data)

        loss = data_loss + 1000 * bc_loss + 10 * pde_loss

        validation_loss = loss_data(NN, val_data, val_label)

        writer.add_scalars('1_loss', {'train': loss,
                                      'validation': validation_loss,
                                      'data_loss': data_loss,
                                      'PDE_loss': pde_loss,
                                      'bc_loss': bc_loss}, iter)

        loss.backward()
        opt.step()

        if store and iter % 50 == 0:
            state = {'model': NN.state_dict(),
                     'optimizer': opt.state_dict(),
                     'epoch': epoch}
            torch.save(state, module_name + '_rec')

        if validation_loss.item() < min_loss:
            min_loss = validation_loss.item()
            print('______Best loss model %.16f______' % loss.item())
            print('PDE loss is %.16f' % pde_loss.item())
            print('DATA loss is %.16f' % data_loss.item())
            print('Validation loss is %.16f' % validation_loss.item())
            if store:
                state = {'model': NN.state_dict(),
                         'optimizer': opt.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name)

        print('------%d epoch------' % epoch)
