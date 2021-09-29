import numpy as np
import matplotlib.pyplot as plt

import initsystems

n = int(input('n = ?'))
try:
    R = float(input('tube R = ?'))
except:
    R = None

try:
    L = float(input('L = ?'))
except:
    L = None

try:
    R_sphere = float(input('sphere R = ?'))
except:
    R_sphere = None

# test tube-grid
x, p, q = initsystems.init_tube_grid(n, R, L)
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(projection='3d')

ax.scatter(x[:,0], x[:,1], x[:,2])

plt.savefig('test_figs/test-tube-grid.pdf')

# test tube-random
x, p, q = initsystems.init_tube(n, R, L)
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(projection='3d')

ax.scatter(x[:,0], x[:,1], x[:,2])

plt.savefig('test_figs/test-tube.pdf')

# test sphere
x, p, q = initsystems.init_sphere(n, R_sphere)
fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(projection='3d')

ax.scatter(x[:,0], x[:,1], x[:,2])

plt.savefig('test_figs/test-sphere.pdf')
