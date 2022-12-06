import os
import torch
from num_af import loss_pde, loss_data, dataset, loss_fpde, loss_interface
from module import AF_module as Module
from parameter import module_name, device, EPOCH, LOSS, sparse_af, LR, ITERATION
import numpy as np
from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
Filter = False
path = 'train_history/af/' + str(sparse_af) + ('p/faf' if Filter else 'p/af')
module_name = path + '/arterial_flow'

if __name__ == '__main__':
    print(device)
    NN = Module().to(device)
    opt = torch.optim.Adam(params=NN.parameters(), lr=LR)

    start_epoch = 0

    writer = SummaryWriter(path)

    (data, label, test_data, test_label, collocation, interface) = dataset

    if os.path.exists(module_name + 'af_rec') and load:
        state = torch.load(module_name + 'af_rec')

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

        pde_loss = []
        for i, x in zip([1, 2, 3], collocation):
            if Filter is True:
                pde_loss.append(loss_fpde(NN, x, i))
            else:
                pde_loss.append(loss_pde(NN, x, i))
        pde_loss = sum(pde_loss)

        data_loss = []
        for i, x, y in zip([1, 2, 3, 3], data, label):
            data_loss.append(loss_data(NN, x, y, i))
        data_loss = sum(data_loss)

        interface_loss = []
        for x in interface:
            interface_loss.append(loss_interface(NN, x))
        interface_loss = sum(interface_loss)

        # sparse_loss = loss_data(NN, sparse_x, sparse_y, 3)

        loss = 1e1 * pde_loss + data_loss + 1e2 * interface_loss  # + sparse_loss

        validation_loss = loss_data(NN, test_data, test_label, 3)

        writer.add_scalars('1_loss', {'train': loss,
                                      'validation': validation_loss,
                                      'data_loss': data_loss,
                                      'PDE_loss': pde_loss,
                                      'Interface_loss': interface_loss}, iter)

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
            print('INTERFACE loss is %.16f' % interface_loss.item())
            print('Validation loss is %.16f' % validation_loss.item())
            if store:
                state = {'model': NN.state_dict(),
                         'optimizer': opt.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name)

        print('------%d epoch------' % epoch)
