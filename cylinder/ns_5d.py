import os
import torch
from num_5d import loss_pde, loss_les, loss_data, loss_icbc
from num_5d import noisy_dataloader_2 as dataloader, v_data, v_label, my_shuffle
from module import ResLinear
from parameter import NN_SIZE_3D, module_name, device, EPOCH, LOSS, LR, ITERATION
import numpy as np
from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
path = '../train_history/5d/ns'
module_name = path + '/' + module_name

if __name__ == '__main__':
    print(device)
    NN = ResLinear(shape=[5, 20, 4])
    # NN_ns = Module(NN_SIZE_3D)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     NN_les = torch.nn.DataParallel(NN_les)
    #     NN_ns = torch.nn.DataParallel(NN_ns)


    NN.to(device)

    opt = torch.optim.Adam(params=NN.parameters(), lr=LR)
    # opt_ns = torch.optim.Adam(params=NN_ns.parameters(), lr=5e-3)

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

    # tensorboard --logdir=/Users/jonathan/Documents/PycharmProjects/cylinder_flow/train_history --port 14514

    for epoch in range(start_epoch, EPOCH):
        for t_x_y, num_solution in dataloader:
            t_x_y = t_x_y.requires_grad_(True).to(device)
            num_solution = num_solution.to(device)
            validation_data, validation_label = my_shuffle(v_data, v_label, 20000)
            validation_data = torch.FloatTensor(validation_data).requires_grad_(False).to(device)
            validation_label = torch.FloatTensor(validation_label).requires_grad_(False).to(device)

            opt.zero_grad()
            iter += 1

            # index = torch.LongTensor(np.random.choice(collocation_size, BATCH, replace=False))
            # t_x_y_col = torch.index_select(collocation_points, 0, index)

            loss_u, loss_v, loss_w, loss_div = loss_pde(NN, t_x_y)  # + loss_les(NN_les, t_x_y_col)
            pde_loss = loss_u + loss_v + loss_w + loss_div

            data_loss_1, data_loss_2 = loss_data(NN, t_x_y, num_solution)
            data_loss = data_loss_1 + data_loss_2

            icbc_loss = loss_icbc(NN)

            loss = data_loss + pde_loss + icbc_loss

            validation_out = NN(validation_data)
            [va_u, va_v, va_w, va_p] = [LOSS(validation_out[:, 0], validation_label[:, 0]) / 4,
                                        LOSS(validation_out[:, 1], validation_label[:, 1]) / 4,
                                        LOSS(validation_out[:, 2], validation_label[:, 2]) / 4,
                                        LOSS(validation_out[:, 3], validation_label[:, 3]) / 4]

            validation_loss = va_u + va_v + va_w + va_p

            writer.add_scalars('1_loss', {'train': loss,
                                          'validation': validation_loss,
                                          'data_loss': data_loss_1,
                                          'std_loss': data_loss_2,
                                          'icbc_loss': icbc_loss}, iter)
            # writer.add_scalars('2_loss', {'train': loss_2,
            #                               'validation': validation_loss_2,
            #                               'data_loss': data_loss_2}, iter)

            writer.add_scalars('pde_loss', {'loss': pde_loss,
                                            'loss_u': loss_u,
                                            'loss_v': loss_v,
                                            'loss_w': loss_w,
                                            'loss_div': loss_div}, iter)

            writer.add_scalars('validation_loss', {'total': validation_loss,
                                                   'u': va_u, 'v': va_v,
                                                   'w': va_w, 'p': va_p}, iter)
            # writer.add_scalars('ns_validation_loss', {'total': validation_loss_2,
            #                                           'u': ns_u, 'v': ns_v,
            #                                           'w': ns_w, 'p': ns_p}, iter)

            loss.backward()  #
            opt.step()
            # loss_2.backward()
            # opt_ns.step()

            if store and iter % 50 == 0:
                state = {'model': NN.state_dict(),
                         'optimizer': opt.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name + '_rec')

                # state = {'model': NN_ns.state_dict(),
                #          'optimizer': opt_ns.state_dict(),
                #          'epoch': epoch}
                # torch.save(state, module_name + '_ns_rec')

            if validation_loss.item() < min_loss:
                min_loss = validation_loss.item()
                print('______Best NS loss model %.8f______' % loss.item())
                print('NS loss is %.8f' % pde_loss.item())
                print('DATA loss is %.8f' % data_loss_1.item())
                print('Validation loss is %.8f' % validation_loss.item())
                # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
                if store:
                    state = {'model': NN.state_dict(),
                             'optimizer': opt.state_dict(),
                             'epoch': epoch}
                    torch.save(state, module_name)

            # if validation_loss_2.item() < min_loss_2:
            #     min_loss_2 = validation_loss_2.item()
            #     print('______Best NS loss model %.8f______' % loss_2.item())
            #     print('PDE loss is %.8f' % pde_loss.item())
            #     print('DATA loss is %.8f' % data_loss_2.item())
            #     print('Validation loss is %.8f' % validation_loss_2.item())
            #     # print('***** Lr = %.8f *****' % opt.state_dict()['param_groups'][0]['lr'])
            #     if store:
            #         state = {'model': NN_ns.state_dict(),
            #                  'optimizer': opt_ns.state_dict(),
            #                  'epoch': epoch}
            #         torch.save(state, module_name + '_ns')

            if iter > ITERATION:
                break

        print('------%d epoch------' % epoch)
        if iter > ITERATION:
            break
