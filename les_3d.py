import os
import torch
from num_3d import loss_pde, loss_les, loss_data
from num_3d import noisy_dataloader as dataloader, v_data, v_label, my_shuffle
from module import Module
from parameter import NN_SIZE_3D, module_name, device, EPOCH, LOSS, collocation_size, BATCH, ITERATION
import numpy as np
from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
module_name = 'train_history/3d/' + module_name + '_3d'

if __name__ == '__main__':
    print(device)

    NN_les = Module(NN_SIZE_3D)
    NN_ns = Module(NN_SIZE_3D)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        NN_les = torch.nn.DataParallel(NN_les)
        NN_ns = torch.nn.DataParallel(NN_ns)

    NN_les.to(device)
    NN_ns.to(device)

    opt_les = torch.optim.Adam(params=NN_les.parameters(), lr=3e-3)
    opt_ns = torch.optim.Adam(params=NN_ns.parameters(), lr=3e-3)

    start_epoch = 0

    writer = SummaryWriter('train_history/3d')

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
            t_x_y = t_x_y.requires_grad_(True).to(device)
            num_solution = num_solution.to(device)
            validation_data, validation_label = my_shuffle(v_data, v_label, 20000)
            validation_data, validation_label = torch.FloatTensor(validation_data).to(device), torch.FloatTensor(
                validation_label).to(device)

            opt_les.zero_grad()
            opt_ns.zero_grad()
            iter += 1
            print(iter)

            # index = torch.LongTensor(np.random.choice(collocation_size, BATCH, replace=False))
            # t_x_y_col = torch.index_select(collocation_points, 0, index)

            les_loss_u, les_loss_v, les_loss_w, les_loss_div = loss_les(NN_les, t_x_y)  # + loss_les(NN_les, t_x_y_col)
            les_loss = les_loss_u + les_loss_v + les_loss_w + les_loss_div
            data_loss_1 = loss_data(NN_les, t_x_y, num_solution)

            pde_loss_u, pde_loss_v, pde_loss_w, pde_loss_div = loss_pde(NN_ns, t_x_y)
            pde_loss = pde_loss_u + pde_loss_v + pde_loss_w + pde_loss_div
            data_loss_2 = loss_data(NN_ns, t_x_y, num_solution)

            loss_1 = data_loss_1 + les_loss
            loss_2 = data_loss_2 + pde_loss

            les_validation_out = NN_les(validation_data)
            [les_u, les_v, les_w, les_p] = [LOSS(les_validation_out[:, 0], validation_label[:, 0]) / 4,
                                            LOSS(les_validation_out[:, 1], validation_label[:, 1]) / 4,
                                            LOSS(les_validation_out[:, 2], validation_label[:, 2]) / 4,
                                            LOSS(les_validation_out[:, 3], validation_label[:, 3]) / 4]

            ns_validation_out = NN_ns(validation_data)
            [ns_u, ns_v, ns_w, ns_p] = [LOSS(ns_validation_out[:, 0], validation_label[:, 0]) / 4,
                                        LOSS(ns_validation_out[:, 1], validation_label[:, 1]) / 4,
                                        LOSS(ns_validation_out[:, 2], validation_label[:, 2]) / 4,
                                        LOSS(ns_validation_out[:, 3], validation_label[:, 3]) / 4]

            validation_loss_1 = les_u + les_v + les_w + les_p
            validation_loss_2 = ns_u + ns_v + ns_w + ns_p

            writer.add_scalars('1_loss', {'train': loss_1,
                                          'validation': validation_loss_1,
                                          'data_loss': data_loss_1}, iter)
            writer.add_scalars('2_loss', {'train': loss_2,
                                          'validation': validation_loss_2,
                                          'data_loss': data_loss_2}, iter)

            writer.add_scalars('les_loss', {'loss': les_loss,
                                            'loss_u': les_loss_u,
                                            'loss_v': les_loss_v,
                                            'loss_w': les_loss_w,
                                            'loss_div': les_loss_div}, iter)
            writer.add_scalars('pde_loss', {'loss': pde_loss,
                                            'loss_u': pde_loss_u,
                                            'loss_v': pde_loss_v,
                                            'loss_w': pde_loss_w,
                                            'loss_div': pde_loss_div}, iter)

            writer.add_scalars('les_validation_loss', {'total': validation_loss_1,
                                                       'u': les_u, 'v': les_v,
                                                       'w': les_w, 'p': les_p}, iter)
            writer.add_scalars('ns_validation_loss', {'total': validation_loss_2,
                                                      'u': ns_u, 'v': ns_v,
                                                      'w': ns_w, 'p': ns_p}, iter)

            loss_1.backward(retain_graph=True)  #
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
                print('Validation loss is %.8f' % validation_loss_1.item())
                # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
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
                print('Validation loss is %.8f' % validation_loss_2.item())
                # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
                if store:
                    state = {'model': NN_ns.state_dict(),
                             'optimizer': opt_ns.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name + '_ns')

        print('------%d epoch------' % epoch)
#        if iter > ITERATION:
#            break
