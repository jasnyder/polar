"""
This file should do what Ala suggested:
    - take a 2d patch of cells, turn on AB polarity coupling and not PCP. let it relax until it's stationary
    - save that configuration to use as initial condition later
    - designate one cell in the middle as a WNT cell
    - turn off PCP interactions of the immediate neighbors of the WNT cell
    - run the shit forward!
"""
import numpy as np
import time
import itertools
import os
from polarcore import Polar, PolarWNT
from initsystems import init_plane
import potentials
import potentials_wnt
import pickle
import torch

# simulation parameters
save_name = 'patch-with-wnt'
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
    with open(f'data/ic/{save_name}.pkl', 'rb') as fobj:
        data, kwargs = pickle.load(fobj)
        x0, p0, q0, lam = data[-1]
except:
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

# now set up the WNT at the center of the cell sheet, and turn on the other coupling terms.

# find the cell closest to the center and make it the WNT cell
index = np.argmin(np.sum(x0**2, axis=1))
wnt_cells = [index]
wnt_threshold = 1e-2

# find the next ring of cells - the immediate neighbors of the WNT cell
sim.find_potential_neighbours()
sim.find_true_neighbours()
ring = sim.idx[index].cpu()

# set lambda parameters - here the WNT cell and its neighbors should not interact via PCP (i.e. lam2 and lam3 are zero)
lam_WNT = np.array([0, 0.5, 0.2, 0.1, 0.2])
lam = np.repeat(lam_WNT[None, :], len(x), axis=0)
lam[index] = lam_relax
lam[ring] = lam_relax


# set division rate and decay-of-division-rate rate
beta[index] = 0.025
beta_decay = 0.5

# Simulation parameters
timesteps = 200
yield_every = 500   # save simulation state every x time steps
diffuse_every = 1
dt = 0.1
eta = 5e-3

# align the q field initially becuase if not, the patch will just fall apart into a weird string that loops on itself
# q0 = np.repeat(np.array([[0., 0., 1.]]), len(q0), axis=0)
# q0 += np.mean(p0) * 0.1

# q0 = np.cross(x0, np.array([-1., 0., 0.]))

# set the potential
potential = potentials_wnt.potential_nematic

sim = PolarWNT(wnt_cells, wnt_threshold, x0, p0, q0, lam, beta, eta=eta, yield_every=yield_every, device="cuda", init_k=50, beta_decay=beta_decay,
               divide_single=True)

"""
Now I'm going to get WNT to diffuse around from the center cell, then turn on the dynamics and see if the WNT is sufficient to get PCP to spiral
"""

def division_decider(sim, tstep):
    """
    This is a function that decides whether or not to let the cells divide

    Idea: take a sublinear function of time, and allow cell division whenever the value of that function passes an integer
    This will make cell division happen more rarely as the simulation progresses.
    """
    T = sim.dt * tstep
    if T < 100 or len(sim.x) > max_cells - 1:
        return False

    def f(T): return 0.05*T
    if int(f(T)) > int(f(T-sim.dt)):
        return True
    else:
        return False


# run!
runner = sim.simulation(potential=potential,
                        division_decider=division_decider)

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
