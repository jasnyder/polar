"""
This file will simulate a sphere of cells with WNT and PCP, gets it to a certain point, then switches the WNT interaction off
"""
import numpy as np
import time
import itertools
import os
from polarcore import PolarWNT
from initsystems import init_random_system, init_tube, init_tube_grid
import potentials_wnt
import pickle
import torch


save_name = 'sphere-wnt-switch-off'
max_cells = 3000

# Grab sphere initial condition from log
n = 1000
with open(f'data/ic/relaxed-sphere-n-{n}.pkl', 'rb') as fobj:
    x, p, q = pickle.load(fobj)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, .35, .225, .075, 0.35])
lam = lam_0
eta = 1e-2  # noise

# Pick some number of cells to be WNT cells and make them divide
n_wnt = 3
index = np.random.randint(len(x), size=n_wnt)
lam = np.repeat(lam[None, :], len(x), axis=0)

# update beta (division rate) parameters and beta_decay: factor by which daughter cells have beta reduced
beta[index] = 1
beta_decay = 0

wnt_cells = index
wnt_threshold = 1e-1
diffuse_every = 1

# Simulation parameters
timesteps = 300
yield_every = 500   # save simulation state every x time steps
switch_wnt_off_at = 225
dt = 0.1

# Potential
potential = potentials_wnt.potential_nematic


def division_decider(sim, tstep):
    """
    This is a function that decides whether or not to let the cells divide

    Idea: take a sublinear function of time, and allow cell division whenever the value of that function passes an integer
    This will make cell division happen more rarely as the simulation progresses.
    """
    T = sim.dt * tstep
    if T < 1000 or len(sim.x) > max_cells - 1:
        return False

    def f(T): return 0.15*T
    if int(f(T)) > int(f(T-sim.dt)):
        return True
    else:
        return False


# Make the simulation runner object:
sim = PolarWNT(x, p, q, lam, beta, wnt_cells=wnt_cells, wnt_threshold=wnt_threshold, eta=eta, yield_every=yield_every,
               device="cuda", init_k=50, beta_decay=beta_decay, divide_single=True)
sim.find_potential_neighbours()
sim.find_true_neighbours()
n_diffuse = 10
for i in range(n_diffuse):
    sim.get_gradient_averaging()
runner = sim.simulation(potential=potential,
                        division_decider=division_decider)

# Running the simulation
data = []  # For storing data
i = 0
t1 = time.time()
print('Starting')

for line in itertools.islice(runner, timesteps):
    i += 1
    print(
        f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(line[0])} cells)')
    data.append(line)
    if i > 0 and i % diffuse_every == 0:
        sim.get_gradient_averaging()
    if i == switch_wnt_off_at:
        lam_0[4] = 0
        print('switching WNT off')
        sim.lam = torch.tensor(np.repeat(lam_0[None, :], len(
            sim.x), axis=0), dtype=sim.dtype, device=sim.device)

    if len(line[0]) > max_cells:
        print('Stopping')
        break

try:
    os.mkdir('data')
except:
    pass
with open(f'data/{save_name}-n-{n}-n_wnt-{n_wnt}-{time.strftime("%d%b%Y-%H-%M-%S")}.pkl', 'wb') as f:
    pickle.dump([data, sim.__dict__], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
