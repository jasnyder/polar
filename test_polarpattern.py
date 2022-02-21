"""
This file is meant to run simulations of the patterning mechanism encoded in polarpattern.py

to do this, I need to set all parameters needed to spin up an instance of PolarWNT, as well as parameters specific to the ligand diffusion dynamics.
"""
import faulthandler

faulthandler.enable()

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


save_name = 'test-polarpattern'
max_cells = 10000

# Grab sphere initial condition from log
n = 500
with open(f'data/ic/relaxed-sphere-n-{n}.pkl', 'rb') as fobj:
    x, p, q = pickle.load(fobj)

beta = 0 + np.zeros(len(x))  # cell division rate
lam_0 = np.array([0.0, .55, .35, .1, 0.5])
lam = lam_0
eta = 1e-2  # noise

# Pick some number of cells to be WNT cells and make them divide
n_wnt = 0
index = np.random.randint(len(x), size=n_wnt)
lam = np.repeat(lam[None, :], len(x), axis=0)

# update beta (division rate) parameters and beta_decay: factor by which daughter cells have beta reduced
beta[index] = 1
beta_decay = 0

wnt_cells = index
wnt_threshold = 1e-1
diffuse_every = 100
diffuse_multiple = 1
wnt_decay = 0
R_decay = -1e-2

# Simulation parameters
timesteps = 50
yield_every = 2000   # save simulation state every x time steps
dt = 0.1

# Potential
potential = potentials_wnt.potential_nematic_reweight

# parameters relating to the ligand diffusion
bounding_radius_factor = 1.5
ligand_step = 0.25
contact_radius = 1
N_ligand = 50000
random_walk_multiple = 5
absoprtion_probability_slope = 1


def division_decider(sim, tstep):
    """
    This is a function that decides whether or not to let the cells divide

    Idea: take a sublinear function of time, and allow cell division whenever the value of that function passes an integer
    This will make cell division happen more rarely as the simulation progresses.
    """
    T = sim.dt * tstep
    if T < 1000 or len(sim.x) > max_cells - 1:
        return False

    def f(T): return 0.1*T
    if int(f(T)) > int(f(T-sim.dt)):
        return True
    else:
        return False


# Make the simulation runner object:
sim = PolarPattern(x, p, q, lam, beta,
                   wnt_cells=wnt_cells, wnt_threshold=wnt_threshold, wnt_decay=wnt_decay,
                   eta=eta, yield_every=yield_every,
                   N_ligand=N_ligand,
                   device="cuda", init_k=50, beta_decay=beta_decay, divide_single=True, absorption_probability_slope=absoprtion_probability_slope,
                   R_decay = R_decay,
                   bounding_radius_factor=bounding_radius_factor, contact_radius=contact_radius, ligand_step=ligand_step, selfnormalizing_absorption_probability=True)

runner = sim.simulation(potential=potential,
                        better_WNT_gradient=True,
                        division_decider=division_decider,
                        random_walk_multiple=random_walk_multiple,
                        yield_ligand=True,
                        wnt_func_of_R=True,
                        beta_func_of_w=True,
                        diffuse_every=diffuse_every,
                        diffuse_multiple=diffuse_multiple)

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

    if len(line[0]) > max_cells:
        print('Stopping')
        break

    with open(f'data/{save_name}-n-{n}-n_wnt-{n_wnt}-in-progress.pkl', 'wb') as f:
        pickle.dump([data, sim.__dict__], f)


try:
    os.mkdir('data')
except:
    pass
with open(f'data/{save_name}-n-{n}-n_wnt-{n_wnt}-{time.strftime("%d%b%Y-%H-%M-%S")}.pkl', 'wb') as f:
    pickle.dump([data, sim.__dict__], f)

print(f'Simulation done, saved {timesteps} datapoints')
print('Took', time.time() - t1, 'seconds')
