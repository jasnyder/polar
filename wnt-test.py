"""
This file is to implement a stripped-down simulation with the WNT added to see if it works as I think it should
"""

import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_grid
import potentials
import pickle

save_name = 'wnt-test'
max_cells = 1000

# initialize a small grid, say 10x10
n = 10
m = 10
x, p, q = init_grid(n, m)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, .5, 0.4, .08, 0.25])
lam = lam_0
eta = 3e-2 # noise

# Make two cells polar and divide them faster
index = np.argmin(np.linalg.norm(x, axis = 1))
lam = np.repeat(lam[None, :], len(x), axis=0)

beta[index] = 0.025
beta_decay = 0

wnt_cells = [index]
wnt_range = 5

# Simulation parameters
timesteps = 100
yield_every = 100   # save simulation state every x time steps
dt = 0.1

# Potential
potential = potentials.potential_wnt_nematic

# cell division rate
division_decider = lambda *args : False

# Make the simulation runner object:
sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50, beta_decay = beta_decay, divide_single = True, wnt_cells = wnt_cells, wnt_range=wnt_range)
runner = sim.simulation(potential=potential, division_decider=division_decider)

# Running the simulation
data = [(x, p, q, lam)]  # For storing data
i = 0
t1 = time.time()
print('Starting')

for xx, pp, qq, lam in itertools.islice(runner, timesteps):
    i += 1
    print(f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(xx)} cells)')
    data.append((xx, pp, qq, lam))

    if len(xx) > max_cells:
        print('Stopping')
        break

try:
    os.mkdir('data')
except:
    pass
with open('data/'+save_name+'.pkl', 'wb') as f:
    pickle.dump([data, sim.__dict__], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')