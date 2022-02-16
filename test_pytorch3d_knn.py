"""
This file is to try out pytorch3d's KNN algorithm. What I really want is an equivalent to scipy.spatial.cKDTree.query_ball_point
"""

import numpy as np
import scipy as sp
import time
from scipy.spatial import cKDTree
import torch
import pytorch3d
from pytorch3d import ops

N1 = 100000
N2 = 1000
D = 3
ligand = torch.randn((1, N1, D), device='cuda') * 3
cells = torch.randn((1, N2, D), device='cuda')
cells /= cells.norm(dim=1, keepdim=True)


# time the pytorch3d method
s = time.perf_counter()
dists, idx, nn = pytorch3d.ops.ball_query(ligand, cells, radius = 1, K=N2)
print(f'pytorch3d, ligand -> cells done in {time.perf_counter() - s} seconds')

s = time.perf_counter()
dists, idx, nn = pytorch3d.ops.ball_query(cells, ligand, radius = 1, K=N1)
print(f'pytorch3d, cells -> ligand done in {time.perf_counter() - s} seconds')

# time the cKDTree method
x = ligand.detach().to('cpu').numpy()[0,:,:]
y = cells.detach().to('cpu').numpy()[0,:,:]

s = time.perf_counter()
tree = cKDTree(x)
inds = tree.query_ball_point(y, 1)
print(f'scipy, ligand -> cells done in {time.perf_counter() - s} seconds')

s = time.perf_counter()
tree = cKDTree(y)
inds = tree.query_ball_point(x, 1)
print(f'scipy, cells -> ligand done in {time.perf_counter() - s} seconds')