from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

initial = [[0, x, n] for x in np.arange(0, 1750, 50) for n in [14000, 16000, 18000, 20000]]
collocation = [[t, x, n] for t in np.arange(0, 48, 1) for x in np.arange(0, 1750, 50) for n in [14000, 20000]]

initial = np.array(initial)
collocation = np.array(collocation)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection="3d")

xi = initial[:, 2]
yi = initial[:, 1]
zi = initial[:, 0]

xc = collocation[:, 2]
yc = collocation[:, 1]
zc = collocation[:, 0]

ax.scatter(xc, yc, zc, c='blue', marker=".", s=20, label='collocation point')
ax.scatter(xi, yi, zi, c='red', marker="*", s=80, label='inital point')

ax.set(xlabel="initial\nnumber ", ylabel="location", zlabel="time")

ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.zaxis.label.set_size(12)

ax.xaxis.set_major_locator(MultipleLocator(2000))
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.9, 1.2, 0.8, 1]))
ax.view_init(32, -32)

plt.legend()
plt.show()
