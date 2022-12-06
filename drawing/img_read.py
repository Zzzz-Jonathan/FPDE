from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard(path, i, j, k):  # path为tensoboard文件的路径
    ea = event_accumulator.EventAccumulator(path)  # 初始化EventAccumulator对象
    ea.Reload()  # 将事件的内容都导进去
    # print(ea.Tags())
    for x in range(i, j, k):
        img = ea.Images(str(x))
        filename = '/Users/jonathan/Documents/PycharmProjects/cylinder_flow/drawing/output/ns/' + str(x) + '.png'
        with open(filename, 'wb') as f:
            f.write(img[0].encoded_image_string)
    # return ea.scalars.Items(ea.scalars.Keys()[0])


if __name__ == '__main__':
    read_tensorboard('../train_history/sparse/15/ns', 2000, 10000, 200)
