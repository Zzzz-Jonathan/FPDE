import os
import torch
from num_4d import loss_pde, loss_les, loss_data, loss_icbc
from num_4d import dataloader_2 as dataloader, v_data, v_label, my_shuffle, norm_para
from module import ResLinear
from parameter import module_name, device, EPOCH, LOSS, ITERATION, LR, Re_4d as RE
from torch.utils.tensorboard import SummaryWriter

load = False
store = True
torch.manual_seed(3407)
path = 'train_history/4d/' + str(RE) + '/ns'
module_name = path + '/' + module_name

if __name__ == '__main__':
    print(device)
    NN = ResLinear(shape=[4, 20, 4]).to(device)

    opt = torch.optim.Adam(params=NN.parameters(), lr=LR)

    start_epoch = 0

    writer = SummaryWriter(path)

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
            validation_label = validation_label * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]

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
            validation_out = validation_out * (norm_para[:, 0] - norm_para[:, 1]) + norm_para[:, 1]
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

            writer.add_scalars('pde_loss', {'loss': pde_loss,
                                            'loss_u': loss_u,
                                            'loss_v': loss_v,
                                            'loss_w': loss_w,
                                            'loss_div': loss_div}, iter)

            writer.add_scalars('validation_loss', {'total': validation_loss,
                                                   'u': va_u, 'v': va_v,
                                                   'w': va_w, 'p': va_p}, iter)

            loss.backward()  #
            opt.step()

            if store and iter % 50 == 0:
                state = {'model': NN.state_dict(),
                         'optimizer': opt.state_dict(),
                         'epoch': epoch}
                torch.save(state, module_name + '_rec')

            if validation_loss.item() < min_loss:
                min_loss = validation_loss.item()
                print('______Best LES loss model %.8f______' % loss.item())
                print('LES loss is %.8f' % pde_loss.item())
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
