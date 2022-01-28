"""
This file defines a class that will simulate the polarity model coupled together with a ligand-receptor type reaction that will create patterns of WNT expression on the growing organoid that will determine where tubes grow from.


"""
from zmq import device
import polarcore
import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree

class PolarPattern(polarcore.PolarWNT):
    def __init__(self, *args, N_ligand = 50000, ligand_step = 1, bounding_radius_factor = 2, contact_radius = 1, R_init = 1.0, R_upregulate = 1e-2, w_upregulate = 1e-2, **kwargs):
        self.N_ligand = N_ligand
        self.ligand_step = ligand_step
        self.bounding_radius_factor = bounding_radius_factor
        self.contact_radius = contact_radius
        super().__init__(*args, **kwargs)
        self.R = torch.ones_like(self.w) * R_init
        self.R_init = R_init
        self.R_upregulate = R_upregulate
        self.w_upregulate = w_upregulate
        self.get_bounding_sphere()
        self.initialize_ligand()

    def get_bounding_sphere(self):
        self.bounding_sphere_center = self.x.mean(dim=0)
        self.bounding_sphere_radius = self.bounding_radius_factor * torch.max(torch.norm(self.x - self.bounding_sphere_center, dim=1))
        return self.bounding_sphere_center, self.bounding_sphere_radius

    def initialize_ligand(self):
        # place ligand particles uniformly at random on the surface of the bouding sphere
        self.get_bounding_sphere()
        pos = torch.randn((self.N_ligand, 3), device = self.device, dtype = self.dtype)
        pos /= torch.norm(pos, dim = 1, keepdim=True)
        ligand = self.bounding_sphere_center + self.bounding_sphere_radius * pos
        self.ligand = ligand.detach()

    def replenish_ligand_particles(self):
        self.get_bounding_sphere()
        n_new = self.N_ligand - len(self.ligand)
        pos = torch.randn((n_new, 3), device = self.device, dtype = self.dtype)
        pos/= torch.norm(pos, dim=1, keepdim=True)
        self.ligand = torch.cat((self.ligand, self.bounding_sphere_center + self.bounding_sphere_radius * pos), dim = 0)


    def random_walk_ligand(self):
        """
        This is the main routine that updates the positions of the ligand particles. What it does is:
        - proposes a random displacement vector for each particle
        - checks if that displacement would take the particle out of the bounding sphere, and if so, reflects it back inward
        - handles collision between ligand particles and cells. Two possibilities:
            - particle is absorbed, with a probability depending on its level of receptor
            - otherwise, particle is reflected out of the cell sheet
        """
        with torch.no_grad():
            # step in a random direction
            dx = torch.empty_like(self.ligand).normal_() * self.sqrt_dt * self.ligand_step
            # hvae a routine that bounces ligand particles off of the surface of the bounding sphere
            dx = self.check_bounding_sphere(dx)
            # have another routine that checks which ligand particles are about to hit a cell, then either absorbs them or reflects them off of the cell
            # this routine may change the number of entries of the ligand and dx tensors!!
            dx = self.handle_ligand_collisions(dx, contact_radius = self.contact_radius)
            # update ligand particle positions
            self.ligand = self.ligand + dx
            # refresh the pool of ligand particles with enough to keep the total number fixed
            self.replenish_ligand_particles()

    def check_bounding_sphere(self, dx):
        # compute by how much each cell would exceed the bounding sphere
        factor = torch.clamp(torch.norm((self.ligand + dx - self.bounding_sphere_center), dim = 1) - self.bounding_sphere_radius, min=0)
        # adjust each row of dx by that factor times the vector pointing towards the center of the sphere
        # factor 2 so that the particle moves _inside_ the sphere rather than to its surface
        dx -= 2* factor[:,None] * (self.ligand - self.bounding_sphere_center)
        return dx


    def absorption_probability(self, R, mean = None, slope = 1):
        if mean is None:
            mean = self.R_init
        return 1+torch.tanh(slope*(R - mean))/2

    def handle_ligand_collisions(self, dx, contact_radius = 1, workers = -1):
        try:
            ligand_tree = cKDTree((self.ligand + dx).detach().to('cpu').numpy())
            inds = ligand_tree.query_ball_point(self.x.detach().to('cpu').numpy(), contact_radius, n_jobs = workers)
        except ValueError:
            import pickle
            with open('data/dump/ligand_tree_fail.pkl','w') as fobj:
                pickle.dump({'ligand':self.ligand.detach().to('cpu')}, fobj)
        
        # len(inds) = len(x) = number of cells
        # inds[i] = list of indices of ligand particles within distance contact_radius of cell i
        absorb = [list(), list()]
        reflect = [list(), list()]
        Rmean = self.R.mean()
        for i, ind in enumerate(inds):
            if ind is None:
                import pickle
                with open('data/dump/ligand_tree_fail.pkl','wb') as fobj:
                    pickle.dump({'ligand':self.ligand.detach().to('cpu')}, fobj)
            # for each cell i
            for j in ind:
                # for each ligand particle near enough to cell i
                # roll a die with success probability depending on R
                if np.random.rand() < self.absorption_probability(self.R[i], mean=Rmean):
                    absorb[0].append(i)
                    absorb[1].append(j)
                else:
                    reflect[0].append(i)
                    reflect[1].append(j)
        # carry out reflection off of cells
        dx = self.reflect(dx, reflect)
        # carry out absorption of ligand by cells
        self.absorb(absorb)
        # for those ligand particles that were absorbed, remove them.
        # removal is accomplished by a boolean mask
        mask = torch.ones(len(dx), dtype = torch.bool, device = self.device)
        mask[absorb[1]] = False
        dx = dx[mask, :]
        self.ligand = self.ligand[mask, :]
        return dx

    def absorb(self, indices):
        self.R[indices[0]] += self.R_upregulate
        self.w[indices[0]] += self.w_upregulate

    def reflect(self, dx, indices):
        # find those ligand particles that would have touched cells, and reflect their dx vector through the plane perpendicular to AB polarity
        # dx[indices[1]] += 2 * torch.abs(torch.sum(dx[indices[1]] * self.p[indices[0]], dim=1, keepdim=True)) * self.p[indices[0]]
        return dx


    def simulation(self, potential, division_decider = lambda *args : True, better_WNT_gradient = False, yield_ligand = False):
        """
        Generator to implement the simulation

        Note: you can interact with this thing. Example:
        ```python
        polarguy = PolarWNT(**)
        sim = polarguy.simulation(***)
        for out in sim:
            data.append(out)
            polarguy.get_gradient_averaging()
        ```

        Parameters
        ----------
            x : torch.Tensor
                Position of each cell in 3D space
            p : torch.Tensor
                AB polarity vector of each cell
            q : torch.Tensor
                PCP vector of each cell
            lam : torch.Tensor
                weights for the terms of the potential function for each cell.
            beta : torch.Tensor
                for each cell, probability of division per unit time
            eta : float
                Strength of the added noise
            potential : callable
                function that computes the value of the potential between two cells, i and j
                call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj)
            yield_every : int, optional
                How many simulation time steps to take between yielding the system state. Default: 1
            dt : float, optional
                Size of the time step. Default: 0.1
            kwargs : dict
                Keyword args passed to self.init_simulation and self.time_step.
                Values passed here override default values
                dt : float
                    time step
                yield_every : int
                    how many time steps to take in between yielding system state

        Yields
        ----------
            x : numpy.ndarray
                Position of each cell in 3D space
            p : numpy.ndarray
                AB polarity vector of each cell
            q : numpy.ndarray
                PCP vector of each cell
            w : numpy.ndarray
                WNT concentration at each cell
            lam : numpy.ndarray
                weights for the terms of the potential function for each cell.
        """

        tstep = 0
        while True:
            tstep += 1

            # perform cell division, depending on the output of the function division_decider
            # by default, always do cell division (this results in exponential growth of number of cells)
            self.random_walk_ligand()

            division = False
            if division_decider(self, tstep):
                if self.divide_single:
                    division = self.cell_division_single()
                else:
                    division = self.cell_division()
            
            n_update = self.update_k(self.true_neighbour_max, tstep)
            self.k = min(self.k, len(self.x) - 1)

            if division or tstep % n_update == 0 or self.idx is None:
                self.find_potential_neighbours()
            
            self.get_gradient_vectors(better=better_WNT_gradient)
            self.gradient_step(tstep, potential=potential)
            self.w = self.w * np.exp(self.dt * self.wnt_decay)

            torch.cuda.empty_cache()

            if tstep % self.yield_every == 0:
                xx = self.x.detach().to("cpu").numpy().copy()
                pp = self.p.detach().to("cpu").numpy().copy()
                qq = self.q.detach().to("cpu").numpy().copy()
                ww = self.w.detach().to("cpu").numpy().copy()
                ll = self.lam.detach().to("cpu").numpy().copy()
                if yield_ligand:
                    lig = self.ligand.detach().to('cpu').numpy().copy()
                    yield xx, pp, qq, ww, ll, lig
                else:
                    yield xx, pp, qq, ww, ll

    def cell_division(self):
        """
        Decides which cells divide, and if they do, places daughter cells.
        If a cell divides, one daughter cell is placed at the same position as the parent cell, and the other is placed one cell diameter away in a uniformly random direction

        Parameters
        ----------
            x : torch.Tensor
                Position of each cell in 3D space
            p : torch.Tensor
                AB polarity vector of each cell
            q : torch.Tensor
                PCP vector of each cell
            lam : torch.Tensor
                weights for the terms of the potential function for each cell.
            beta : torch.Tensor
                for each cell, probability of division per unit time
            dt : float
                Size of the time step.
            kwargs : dict
                Valid keyword arguments:
                beta_decay : float
                    the factor by which beta (probability of cell division per unit time) decays upon cell division.
                    after cell division, one daughter cell has the same beta as the mother (b0), and the other has beta = b0 * beta_decay

        Returns
        ---------
            division : bool
                True if cell division has taken place, otherwise False
            x : torch.Tensor
                Position of each cell in 3D space
            p : torch.Tensor
                AB polarity vector of each cell
            q : torch.Tensor
                PCP vector of each cell
            lam : torch.Tensor
                weights for the terms of the potential function for each cell.
            beta : torch.Tensor
                for each cell, probability of division per unit time
        """
        if torch.sum(self.beta) < self.do_nothing_threshold:
            return False
        
        # set probability according to beta and dt
        d_prob = self.beta * self.dt
        # flip coins
        draw = torch.empty_like(self.beta).uniform_()
        # find successes
        events = draw < d_prob
        division = False

        if torch.sum(events) > 0:
            with torch.no_grad():
                division = True
                # find cells that will divide
                idx = torch.nonzero(events)[:, 0]

                x0 = self.x[idx, :]
                p0 = self.p[idx, :]
                q0 = self.q[idx, :]
                w0 = self.w[idx]
                R0 = self.R[idx]
                l0 = self.lam[idx, :]
                b0 = self.beta[idx] * self.beta_decay

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                # move the cells so that the center of mass of each pair is at the same place as the mother cell was
                x0 = x0 - move/2
                self.x[idx, :] += move/2

                # divide WNT from mother cells evenly to daughter cells
                self.w[idx] /= 2
                w0 /= 2
                self.R[idx] /= 2
                R0 /= 2

                # append new cell data to the system state
                self.x = torch.cat((self.x, x0))
                self.p = torch.cat((self.p, p0))
                self.q = torch.cat((self.q, q0))
                self.w = torch.cat((self.w, w0))
                self.R = torch.cat((self.R, R0))
                self.lam = torch.cat((self.lam, l0))
                self.beta = torch.cat((self.beta, b0))

        # replenish WNT at source cell(s)
        self.w[self.wnt_cells] = 1

        self.x.requires_grad = True
        self.p.requires_grad = True
        self.q.requires_grad = True

        return division

    def cell_division_single(self):
        """
        Selects exactly one cell to divide and divides it.
        If a cell divides, one daughter cell is placed at the same position as the parent cell, and the other is placed one cell diameter away in a uniformly random direction
        """
        if torch.sum(self.beta) < self.do_nothing_threshold:
            return False

        idx = torch.multinomial(self.beta, 1)
        with torch.no_grad():
            x0 = self.x[idx, :]
            p0 = self.p[idx, :]
            q0 = self.q[idx, :]
            w0 = self.w[idx]
            R0 = self.R[idx]
            l0 = self.lam[idx, :]
            b0 = self.beta[idx] * self.beta_decay

            # make a random vector and normalize to get a random direction
            move = torch.empty_like(x0).normal_()
            move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

            # move the cells so that the center of mass of each pair is at the same place as the mother cell was
            x0 = x0 - move/2
            self.x[idx, :] += move/2

            # divide WNT from mother cells evenly to daughter cells
            self.w[idx] /= 2
            w0 /= 2
            self.R[idx] /= 2
            R0 /= 2

            # append new cell data to the system state
            self.x = torch.cat((self.x, x0))
            self.p = torch.cat((self.p, p0))
            self.q = torch.cat((self.q, q0))
            self.w = torch.cat((self.w, w0))
            self.R = torch.cat((self.R, R0))
            self.lam = torch.cat((self.lam, l0))
            self.beta = torch.cat((self.beta, b0))

        # replenish WNT at source cell(s)
        self.w[self.wnt_cells] = 1

        self.x.requires_grad = True
        self.p.requires_grad = True
        self.q.requires_grad = True

        return True
