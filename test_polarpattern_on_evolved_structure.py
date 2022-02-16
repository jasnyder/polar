"""
This file is meant to run simulations of the patterning mechanism encoded in polarpattern.py

to do this, I need to set all parameters needed to spin up an instance of PolarWNT, as well as parameters specific to the ligand diffusion dynamics.
"""
import numpy as np
import time
import itertools
import os
from polarcore import PolarWNT
from polarpattern import PolarPattern
from initsystems import init_random_system, init_tube, init_tube_grid
import potentials_wnt
import pickle
import torch


save_name = 'test-polarpattern-on-evolved'
max_cells = 10000

# grab a configuration (x, p, q) from a previous run
fname = 'data/sphere-wnt-n-1000-n_wnt-5-04Jan2022-14-43-36.pkl'
with open(fname,'rb') as fobj:
    data, kwargs = pickle.load(fobj)
x, p, q, ww, ll = data[-1]
n=len(x)

beta = 0 + np.zeros(len(x))  # cell division rate
eta = kwargs['eta']  # noise

# Pick some number of cells to be WNT cells and make them divide
n_wnt = 0
index = np.random.randint(len(x), size=n_wnt)
lam = ll

# update beta (division rate) parameters and beta_decay: factor by which daughter cells have beta reduced
beta[index] = 1
beta_decay = 0

wnt_cells = index
wnt_threshold = kwargs['wnt_threshold']
diffuse_every = 4
diffuse_multiple = 1
wnt_decay = -5e-5
R_decay = -5e-5

# Simulation parameters
timesteps = 50
yield_every = 500   # save simulation state every x time steps
dt = 0.1

# Potential
potential = potentials_wnt.potential_nematic_reweight

# parameters relating to the ligand diffusion
bounding_radius_factor = 1.5
ligand_step = 0.5
contact_radius = 1
N_ligand = 50000
random_walk_multiple = 2
absoprtion_probability_slope = 4

# Make the simulation runner object:
sim = PolarPattern(x, p, q, lam, beta,
                   wnt_cells=wnt_cells, wnt_threshold=wnt_threshold, wnt_decay=wnt_decay,
                   eta=eta, yield_every=yield_every,
                   N_ligand=N_ligand,
                   device="cuda", init_k=50, beta_decay=beta_decay, divide_single=True, absorption_probability_slope=absoprtion_probability_slope,
                   R_decay = R_decay,
                   bounding_radius_factor=bounding_radius_factor, contact_radius=contact_radius, ligand_step=ligand_step)

runner = sim.simulation(potential=potential,
                        better_WNT_gradient=True,
                        division_decider=lambda *args:False,
                        random_walk_multiple=random_walk_multiple,
                        yield_ligand=True)

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
    sim.beta = sim.w.detach().clone()**4
    if i > 0 and i % diffuse_every == 0:
        for j in range(diffuse_multiple):
            sim.get_gradient_averaging()

    if len(line[0]) > max_cells:
        print('Stopping')
        break

try:
    os.mkdir('data')
except:
    pass
with open(f'data/{save_name}-n-{n}-n_wnt-{n_wnt}-{time.strftime("%d%b%Y-%H-%M-%S")}.pkl', 'wb') as f:
    sim.__dict__.update({'diffuse_every': diffuse_every,
                        'diffuse_multiple': diffuse_multiple})
    pickle.dump([data, sim.__dict__], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
