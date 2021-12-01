"""
This file runs a simulation starting from an initial condition where cells are arranged in a tube, and runs it for a long time to make sure it's in equilibrium.

There is no cell division
"""
import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_sphere
import potentials
import pickle

save_name = 'ic/relaxing-sphere'

n = int(input('n = ? ') or 500)

x, p, q = init_sphere(n)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, 1, 0, 0])
lam = np.repeat(lam_0[None, :], len(x), axis=0)
eta = 1e-3  # noise

# Simulation parameters
timesteps = 20
yield_every = 1000  # save simulation state every x time steps
dt = 0.1

potential = potentials.potential_nematic

# Make the simulation runner object:
sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50,
            divide_single=True)

# do the initial phase: relax the initial condition
runner = sim.simulation(potential=potential,
                        division_decider=lambda *args: False)

# Running the simulation
data = [(x, p, q, lam)]  # For storing data
i = 0
t1 = time.time()
print('Starting')

for xx, pp, qq, lam in itertools.islice(runner, timesteps):
    i += 1
    print(f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(xx)} cells)')
    data.append((xx, pp, qq, lam))

try:
    os.mkdir('data')
except:
    pass
with open('data/'+save_name+'.pkl', 'wb') as f:
    pickle.dump([data, sim.__dict__], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
