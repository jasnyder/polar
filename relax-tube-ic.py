"""
This file runs a simulation starting from an initial condition where cells are arranged in a tube, and runs it for a long time to make sure it's in equilibrium.

There is no cell division
"""
import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_tube_grid
import pickle
import torch

kwargs = dict()
def add_entries(d, var_names):
    # hacky little function to add entries to a dictionary by variable name. I just hate having the variable name written twice in a line
    # can accept either a single string or a list of strings. or, potentially, nested lists that eventually have strings in them
    if type(var_names) is list:
        for var_name in var_names:
            add_entries(d, var_name)
    elif type(var_names) is str:
        d[var_names] = eval(var_names)
    return

save_name = 'ic/tube-ic-along'

# Initialize a tube
n = int(input('n = ? '))
try:
    R = float(input('R = ? '))
except:
    R = None
try:
    L = float(input('L = ? '))
except:
    L = None
comb_direction = input('comb direction? ')
if comb_direction not in ['along', 'around']:
    comb_direction = None
add_entries(kwargs, ['n','L','R','comb_direction'])


x, p, q = init_tube_grid(n, R, L, comb_direction=comb_direction)
jitter = 0
x += jitter*np.random.randn(*x.shape)
p += jitter*np.random.randn(*p.shape)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, 1, 0, 0])
lam = lam_0
eta = 1e-3  # noise
add_entries(kwargs, 'eta')
kwargs['lam_0'] = lam


# Simulation parameters
timesteps = 100
yield_every = 1000  # save simulation state every x time steps
dt = 0.1

add_entries(kwargs, ['timesteps','yield_every', 'dt'])

# Potential
def potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj, **kwargs):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij


# Make the simulation runner object:
sim = Polar(device="cuda", init_k=50)
runner = sim.simulation(x, p, q, lam, beta, potential=potential, **kwargs)

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
    pickle.dump([data, kwargs], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
