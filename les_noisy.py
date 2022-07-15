import os
import torch
from condition import loss_pde, loss_data, loss_les
from data_num import noisy_norm_dataloader as dataloader, collocation_points
from data_num import validation_data, validation_label
from module import Module
from parameter import NN_SIZE, module_name, device, EPOCH, LOSS, collocation_size, BATCH
import numpy as np

from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
module_name = module_name + '_les'

if __name__ == '__main__':
    print(device)
    NN_les = Module(NN_SIZE).to(device)
    opt_les = torch.optim.Adam(params=NN_les.parameters())

    NN_ns = Module(NN_SIZE).to(device)
    opt_ns = torch.optim.Adam(params=NN_ns.parameters())

    start_epoch = 0

    writer = SummaryWriter('/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history')

    if os.path.exists(module_name + '_les_rec') and load:
        state = torch.load(module_name + '_les_rec')

        NN_les.load_state_dict(state['model'])
        opt_les.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

        print('Start from epoch = %d' % start_epoch)

    min_loss_1, min_loss_2 = 1e6, 1e6
    iter = 0

    # tensorboard --logdir=/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history --port 14514

    for epoch in range(start_epoch, EPOCH):
        for t_x_y, num_solution in dataloader:
            opt_les.zero_grad()
            opt_ns.zero_grad()
            iter += 1

            # index = torch.LongTensor(np.random.choice(collocation_size, BATCH, replace=False))
            # t_x_y_col = torch.index_select(collocation_points, 0, index)

            les_loss = loss_les(NN_les, t_x_y)  # + loss_les(NN_les, t_x_y_col)
            data_loss_1 = loss_data(NN_les, t_x_y, num_solution)

            pde_loss = loss_pde(NN_ns, t_x_y)
            data_loss_2 = loss_data(NN_ns, t_x_y, num_solution)

            loss_1 = data_loss_1 + les_loss
            loss_2 = data_loss_2 + pde_loss

            validation_loss_1 = LOSS(NN_les(validation_data), validation_label)
            validation_loss_2 = LOSS(NN_ns(validation_data), validation_label)

            writer.add_scalars('1_loss', {'train': loss_1,
                                          'validation': validation_loss_1,
                                          'data_loss': data_loss_1}, iter)
            writer.add_scalars('2_loss', {'train': loss_2,
                                          'validation': validation_loss_2,
                                          'data_loss': data_loss_2}, iter)
            writer.add_scalar('les_loss', les_loss, iter)
            writer.add_scalar('pde_loss', pde_loss, iter)

            loss_1.backward(retain_graph=True)
            opt_les.step()
            loss_2.backward()
            opt_ns.step()

            if store and iter % 50 == 0:
                state = {'model': NN_les.state_dict(),
                         'optimizer': opt_les.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name + '_les_rec')

                state = {'model': NN_ns.state_dict(),
                         'optimizer': opt_ns.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name + '_ns_rec')

            if validation_loss_1.item() < min_loss_1:
                min_loss_1 = validation_loss_1.item()
                print('______Best LES loss model %.8f______' % loss_1.item())
                print('LES loss is %.8f' % les_loss.item())
                print('DATA loss is %.8f' % data_loss_1.item())

                if store:
                    state = {'model': NN_les.state_dict(),
                             'optimizer': opt_les.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name + '_les')

            if validation_loss_2.item() < min_loss_2:
                min_loss_2 = validation_loss_2.item()
                print('______Best NS loss model %.8f______' % loss_2.item())
                print('PDE loss is %.8f' % pde_loss.item())
                print('DATA loss is %.8f' % data_loss_2.item())

                if store:
                    state = {'model': NN_ns.state_dict(),
                             'optimizer': opt_ns.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name + '_ns')

        print('------%d epoch------' % epoch)
