import numpy as np
import fluidfoam

path = '/Users/jonathan/OpenFOAM/jonathan-v2206/run/cylinder3d_1'
Re = 2 ** 10
starttime = 1
endtime = 3
step = 2


def read_site():
    x = fluidfoam.readscalar(path, '0', 'Cx')[:, None]
    y = fluidfoam.readscalar(path, '0', 'Cy')[:, None]
    z = fluidfoam.readscalar(path, '0', 'Cz')[:, None]
    t = np.arange(starttime, endtime + 1e-5, step / 100)[:, None] - starttime

    np.savez('data/re_expor/site.npz', x=x, y=y, z=z, t=t)


def read_U():
    times = []

    for i in range(starttime, endtime):
        for j in range(10):
            if j == 0:
                times.append(str(i))
            else:
                times.append(str(i) + '.' + str(j))
            for k in range(step, 10, step):
                times.append(str(i) + '.' + str(j) + str(k))
    times.append(str(endtime))

    # print(times)

    us, vs, ws, ps = [], [], [], []
    for i in times:
        U = fluidfoam.readvector(path, i, 'U')
        p = fluidfoam.readscalar(path, i, 'p')[:, None]
        u, v, w = U[0][:, None], U[1][:, None], U[2][:, None]

        us.append(u)
        vs.append(v)
        ws.append(w)
        ps.append(p)

    [us, vs, ws, ps] = [np.array(us), np.array(vs),
                        np.array(ws), np.array(ps)]

    np.savez('data/re_expor/field_%d.npz' % Re, u=us, v=vs, w=ws, p=ps)


if __name__ == '__main__':
    read_U()
    # times = []
    # starttime = 1
    # endtime = 3
    # step = 2
    # for i in range(starttime, endtime):
    #     for j in range(10):
    #         if j == 0:
    #             times.append(str(i))
    #         else:
    #             times.append(str(i) + '.' + str(j))
    #         for k in range(step, 10, step):
    #             times.append(str(i) + '.' + str(j) + str(k))
    # times.append(str(endtime))
    #
    # print(times)
    pass
