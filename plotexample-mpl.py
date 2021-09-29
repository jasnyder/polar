from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import pickle

fname = input('Enter data filename: ')#'data/test1.pkl'

print('Loading data from file '+fname)
with open(fname, 'rb') as f:
    data = pickle.load(f)  # contains x, p, q, lam
# data[t][0] == x, x[i, k] = position of particle i in dimension k
# data[t][1] == p, p[i, k] = AB polarity of particle i in dimension k
# data[t][2] == q, q[i, k] = PCP of particle i in dimension k


t = 0

fig = plt.figure(figsize = (12, 10))
ax = fig.add_subplot(projection='3d', elev = 90)
ax.set_box_aspect(None, zoom = 3)
ax.axis('off')

def update(num, data, line):
    line.set_data(data[num][0][:, :2].T)
    line.set_3d_properties(data[num][0][:, 2])

line, = ax.plot(data[t][0][:, 0], data[t][0][:, 1], data[t][0][:, 2],
                marker = 'o', linestyle = '')


# Setting the axes properties

maxes = [tstep[0].max() for tstep in data]
mins = [tstep[0].min() for tstep in data]
Max = max(maxes)
Min = min(mins)

ax.set_xlim3d([Min, Max])
ax.set_xlabel('X')

ax.set_ylim3d([Min, Max])
ax.set_ylabel('Y')

ax.set_zlim3d([Min, Max])
ax.set_zlabel('Z')

print('Creating animation')
ani = animation.FuncAnimation(fig, update, len(data), fargs=(data, line),interval=10000/len(data), blit=False)
writergif = animation.PillowWriter(fps=30) 
print('Saving animation')
ani.save('animations/'+fname.split('/')[1].split('.')[0]+'.gif', writer=writergif)