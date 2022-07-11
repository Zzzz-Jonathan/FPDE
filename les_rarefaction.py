import os
import torch
from condition import loss_pde, loss_data, loss_les
from data_num import rare_dataloader as dataloader
from data_num import rare_validation_data, rare_validation_label
from module import Module
from parameter import NN_SIZE, module_name, device, EPOCH, LOSS
import numpy as np

from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
module_name = module_name + '_les'

if __name__ == '__main__':
    print(device)
    NN = Module(NN_SIZE).to(device)
    opt = torch.optim.Adam(params=NN.parameters())
    start_epoch = 0

    writer = SummaryWriter('/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history')

    if os.path.exists(module_name) and load:
        state = torch.load(module_name)

        NN.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

        print('Start from epoch = %d' % start_epoch)

    min_loss = 1e6
    iter = 0

    # tensorboard --logdir=/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history --port 14514

    for epoch in range(start_epoch, EPOCH):
        for t_x_y, num_solution in dataloader:
            opt.zero_grad()
            iter += 1

            les_loss = loss_les(NN, t_x_y)
            data_loss = loss_data(NN, t_x_y, num_solution)
            # with the les loss, total loss decrease faster and lower

            loss = data_loss + les_loss

            validation_loss = LOSS(NN(rare_validation_data), rare_validation_label)

            writer.add_scalar('1_loss', {'train': loss, 'validation': validation_loss}, iter)
            writer.add_scalar('3_les_loss', les_loss, iter)
            writer.add_scalar('4_data_loss', data_loss, iter)

            loss.backward()
            opt.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                print('______Best loss model %.8f______' % loss.item())
                print('LES loss is %.8f' % les_loss.item())
                print('DATA loss is %.8f' % data_loss.item())
                # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
                if store:
                    state = {'model': NN.state_dict(),
                             'optimizer': opt.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name)

        print('------%d epoch------' % epoch)
