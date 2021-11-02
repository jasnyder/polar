"""
This file is to run a simulation using the newly-modified code base.

Makes PCP interactions nematic, i.e. not having a preferred direction
"""
import numpy as np
import time
import itertools
import os
from polarcore import Polar
from initsystems import init_random_system, init_tube, init_tube_grid
import pickle
import torch


save_name = 'tube-bulge-nematic'
max_cells = 3000

# Grab tube initial condition from log
with open('data/ic/relaxed-tube-around.pkl', 'rb') as fobj:
    x, p, q = pickle.load(fobj)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, .5, .42, .08])
lam = lam_0
eta = 3e-2 # noise

# Make two cells polar and divide them faster
index = np.argmin(x[:,0])
lam = np.repeat(lam[None, :], len(x), axis=0)
lam_new = (0, .7, .25, .05)
lam[index, :] = lam_0
beta[index] = 0.025
beta_decay = 0

# Simulation parameters
timesteps = 500
yield_every = 200  # save simulation state every x time steps
dt = 0.1

# Potential
def potential(x, d, dx, lam_i, lam_j, pi, pj, qi, qj):
    S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)
    S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.abs(torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2))

    lam1 = 0.5 * (lam_i + lam_j)
    lam2 = lam1.clone()
    lam2[:, : 0] = 1
    lam2[:, :, 1:] = 0
    mask1 = 1 * (lam1[:, :, 0] > 0.5)

    lam = lam1 * (1 - mask1[:, :, None]) + lam2 * mask1[:, :, None]

    S = lam[:, :, 0] + lam[:, :, 1] * S1 + lam[:, :, 2] * S2 + lam[:, :, 3] * S3
    Vij = torch.exp(-d) - S * torch.exp(-d / 5)
    return Vij

def division_decider(sim, tstep):
    """
    This is a function that decides whether or not to let the cells divide
    
    Idea: take a sublinear function of time, and allow cell division whenever the value of that function passes an integer
    This will make cell division happen more rarely as the simulation progresses.
    """
    T = sim.dt * tstep
    if T < 1000 or len(sim.x) > max_cells - 1:
        return False
    f = lambda T : 0.75*T
    if int(f(T)) > int(f(T-sim.dt)):
        return True
    else:
        return False


# Make the simulation runner object:
sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50, beta_decay = beta_decay, divide_single = True)
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
