"""
This file should do what Ala suggested:
    - take a 2d patch of cells, turn on AB polarity coupling and not PCP. let it relax until it's stationary
    - save that configuration to use as initial condition later
    - designate one cell in the middle as a WNT cell
    - turn off PCP interactions of the immediate neighbors of the WNT cell
    - run the shit forward!
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
import itertools
import os
from polarcore import Polar, PolarWNT
from initsystems import init_plane
import plotly.express as px
import potentials
import potentials_wnt
import pickle
import torch

# simulation parameters
save_name = 'patch-with-pcp'
ic_name = 'patch-with-wnt'
max_cells = 1000

# get initial condition
n = 169
L = 5
x, p, q = init_plane(n, L)

beta = 0 + np.zeros(len(x))  # cell division rate
eta = 3e-2  # noise
beta_decay = 0

# initialize lam for the first phase: relaxation
lam_relax = np.array([0, .4, 0, 0, 0])
lam = lam_relax
lam = np.repeat(lam[None, :], len(x), axis=0)


# Simulation parameters
timesteps = 100
yield_every = 500   # save simulation state every x time steps
dt = 0.1

# for the first phase we only want AB polarity interactions, so the normal potential should work just fine
potential = potentials.potential_nematic

sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50, beta_decay=beta_decay,
            divide_single=True)

# do the initial phase: relax the initial condition
runner = sim.simulation(potential=potential,
                        division_decider=lambda *args: False)

try:
    with open(f'data/ic/{ic_name}.pkl', 'rb') as fobj:
        data, kwargs = pickle.load(fobj)
        x0, p0, q0, lam = data[-1]
except FileNotFoundError:
    data = list()  # For storing data
    i = 0
    t1 = time.time()
    print('Starting')

    for xx, pp, qq, lam in itertools.islice(runner, timesteps):
        i += 1
        print(
            f'Relaxing {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(xx)} cells)')
        data.append((xx, pp, qq, lam))

        if len(xx) > max_cells:
            print('Stopping')
            break

    with open(f'data/ic/{save_name}.pkl', 'wb') as fobj:
        pickle.dump([data, sim.__dict__], fobj)
    x0, p0, q0, lam = data[-1]


# set lambda parameters - here I just turn on PCP and try to get it to align with itself and perpendicular to AB
lam_PCP = np.array([0, 0.7, 0.2, 0.1])
lam = np.repeat(lam_PCP[None, :], len(x), axis=0)

potential = potentials.potential_nematic

sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50, beta_decay=beta_decay,
            divide_single=True)

# do the initial phase: relax the initial condition
runner = sim.simulation(potential=potential,
                        division_decider=lambda *args: False)

# Simulation parameters
timesteps = 100
yield_every = 200   # save simulation state every x time steps

data = list()  # For storing data
i = 0
t1 = time.time()
print('Starting')

for line in itertools.islice(runner, timesteps):
    i += 1
    print(
        f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(line[0])} cells)')
    data.append(line)
    # if i>0 and i % diffuse_every == 0:
    #     sim.get_gradient_averaging()

    if len(line[0]) > max_cells:
        print('Stopping')
        break

with open(f'data/{save_name}.pkl', 'wb') as fobj:
    pickle.dump([data, sim.__dict__], fobj)