import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from scipy.interpolate import griddata


def gird(x, y, z):
    dxs = np.linspace(-2.5, 7.5, 200)
    dys = np.linspace(-2.5, 2.5, 100)
    dxs, dys = np.meshgrid(dxs, dys)

    z_new = griddata((x, y), z, (dxs, dys), method='linear')

    return z_new


def fig2data(x, y, z1, z2, z3):
    z1 = gird(x, y, z1)
    z2 = gird(x, y, z2)
    z3 = gird(x, y, z3)

    plt.close('all')
    fig = plt.figure()
    plot1 = fig.add_subplot(221)
    plot2 = fig.add_subplot(222)
    plot3 = fig.add_subplot(223)

    # cylinder = plt.Circle(xy=(250, 250), radius=50, alpha=1, color='white')

    plot1.imshow(z1, cmap=plt.get_cmap('hot'))
    # plot1.add_patch(cylinder)
    plot2.imshow(z2, cmap=plt.get_cmap('hot'))
    # plot2.add_patch(cylinder)
    plot3.imshow(z3, cmap=plt.get_cmap('hot'))
    # plot3.add_patch(cylinder)

    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)

    return image
