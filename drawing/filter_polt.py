import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
x_ = np.linspace(0, 2 * np.pi, 100)


def gx1(_x):
    ans = []
    for i in _x:
        if 1 / 2 - abs(i) > 0:
            ans.append(1)
        else:
            ans.append(0)

    return np.array(ans)


def _gx1(_x):
    ans = []
    for i in _x:
        ans.append(np.sin(i / 2) / (i / 2))

    return np.array(ans)


def gx2(_x):

    return np.sqrt(6 / np.pi) * np.exp(-6 * abs(_x) ** 2)


def _gx2(_x):

    return np.exp(-1 * _x**2 / 24)


def gx3(_x):

    return np.sin(abs(_x) * np.pi) / (abs(_x) * np.pi)


def _gx3(_x):
    ans = []
    xx = np.pi - abs(_x)
    for i in xx:
        if i > 0:
            ans.append(1)
        else:
            ans.append(0)

    return np.array(ans)


if __name__ == '__main__':
    y = gx3(x)
    y_ = _gx3(x_)

    plt.plot(x, y, linewidth=2.5, color='blue', label='G(x)')
    plt.legend(prop={'size': 22})
    plt.show()

    plt.plot(x_, y_, linewidth=2.5, color='red', label='$\hat G(x)$')
    plt.legend(prop={'size': 22})
    plt.show()
