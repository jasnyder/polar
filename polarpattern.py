"""
This file defines a class that will simulate the polarity model coupled together with a ligand-receptor type reaction that will create patterns of WNT expression on the growing organoid that will determine where tubes grow from.


"""
from zmq import device
import polarcore
import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree

class PolarPattern(polarcore.PolarWNT):
    def __init__(self, *args, N_ligand = 50000, ligand_step = 1, bounding_radius_factor = 2, R_init = 1.0, **kwargs):
        self.N_ligand = N_ligand
        self.ligand_step = ligand_step
        self.bounding_radius_factor = bounding_radius_factor
        super().__init__(*args, **kwargs)
        self.R = torch.ones_like(self.w) * R_init
        self.R_init = R_init
        self.get_bounding_sphere()
        self.initialize_ligand()

    def get_bounding_sphere(self):
        self.bounding_sphere_center = self.x.mean(axis=0)
        self.bounding_sphere_radius = self.bounding_radius_factor * torch.sqrt(torch.sum((self.x - self.bounding_sphere_center[:,None])**2))
        return self.bounding_sphere_center, self.bounding_sphere_radius

    def initialize_ligand(self):
        # place ligand particles uniformly at random on the surface of the bouding sphere
        self.get_bounding_sphere()
        pos = torch.randn((self.N_ligand, 3), device = self.device, dtype = self.dtype)
        pos /= torch.norm(pos, dim = 1, keepdim=True)
        self.ligand = self.bounding_sphere_center + self.bounding_sphere_radius * pos
        self.ligand.requires_grad = False

    def random_walk_ligand(self):
        # step in a random direction
        dx = torch.empty_like(self.ligand).normal_() * self.sqrt_dt * self.ligand_step
        # hvae a routine that bounces ligand particles off of the surface of the bounding sphere
        dx = self.check_bounding_sphere(dx)
        # have another routine that checks which ligand particles are about to hit a cell, then either absorbs them or reflects them off of the cell
        # this routine may change the number of entries of the ligand and dx tensors!!
        dx = self.handle_ligand_collisions(dx)
        self.ligand = self.ligand + dx

    def absorption_probability(self, R, mean = None):
        if mean is None:
            mean = self.R_init
        return np.tanh(R - mean)

    def handle_ligand_collisions(self, dx, contact_radius = 1, workers = -1):
        ligand_tree = cKDTree((self.ligand + dx).detach().to('cpu').numpy())
        inds = ligand_tree.query_ball_point(self.x.detach().to('cpu').numpy(), contact_radius, n_jobs = workers)
        # len(inds) = len(x) = number of cells
        # inds[i] = list of indices of ligand particles within distance contact_radius of cell i
        absorb = list()
        reflect = list()
        for i, ind in enumerate(inds):
            # for each cell i
            for j in ind:
                # for each ligand particle near enough to cell i
                # roll a die with success probability depending on R
                if np.random.rand() < np.tanh(self.R[i] - self.R_init):
                    absorb.append((i, j))
                else:
                    reflect.append((i, j))

    def absorb(i, j):



    def spawn_ligand_particles(self):
        self.get_bounding_sphere()
        n_new = self.N_ligand - len(self.ligand)
        pos = torch.randn((n_new, 3), device = self.device, dtype = self.dtype)
        pos/= torch.norm(pos, dim=1, keepdim=True)
        self.ligand = torch.cat((self.ligand, self.bounding_sphere_center + self.bounding_sphere_radius * pos), dim = 0)
