import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree


class Polar:
    """
    Class to define and run simulations of the polarity model of cell movement

    Examples:
    ----------
        ```
        sim = Polar(x, p, q, lam, beta, eta=eta, yield_every=yield_every, init_k=50)
        runner = sim.simulation(potential=potential)

        # Running the simulation
        data = []  # For storing data
        i = 0
        t1 = time.time()
        print('Starting')

        for xx, pp, qq, lam in itertools.islice(runner, timesteps):
            i += 1
            print(f'Running {i} of {timesteps}   ({yield_every * i} of {yield_every * timesteps})   ({len(xx)} cells)')
            data.append((xx, pp, qq, lam))

            if len(xx) > 1000:
                print('Stopping')
                break
        ```
    """
    def __init__(self, x, p, q, lam, beta, eta, yield_every, dt = 0.1, beta_decay = 1.0, do_nothing_threshold = 1e-5, divide_single = False,
                 device='cuda', dtype=torch.float, init_k=100, callback=None):
        self.device = device
        self.dtype = dtype

        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None
        self.callback = callback

        self.x = x
        self.p = p
        self.q = q
        self.beta = beta
        self.dt = dt
        self.sqrt_dt = None
        self.lam = lam

        self.init_simulation()
        self.eta = eta
        self.yield_every = yield_every
        self.beta_decay = beta_decay
        self.do_nothing_threshold = do_nothing_threshold
        self.divide_single = divide_single

    def init_simulation(self):
        """
        Checks input dimensions, cleans and converts into torch.Tensor types

        Parameters
        ----------
            dt : float
                size of time step for simulation
            lam : array
                weights for the terms of the potential function, possibly different for each cell.
            p : array_like
                AB polarity vector of each cell
            q : array_like
                PCP vector of each cell
            x : array_like
                Position of each cell in 3D space
            beta : array_like
                for each cell, probability of division per unit time
            kwargs : dict
                keyword arguments

        Returns
        ----------
            None
        """
        assert len(self.x) == len(self.p)
        assert len(self.q) == len(self.x)
        assert len(self.beta) == len(self.x)

        sqrt_dt = np.sqrt(self.dt)
        x = torch.tensor(self.x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(self.p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(self.q, requires_grad=True, dtype=self.dtype, device=self.device)

        beta = torch.tensor(self.beta, dtype=self.dtype, device=self.device)

        lam = torch.tensor(self.lam, dtype=self.dtype, device=self.device)
        # if lam is not given per-cell, return an expanded view
        if len(lam.shape) == 1:
            lam = lam.expand(x.shape[0], lam.shape[0]).clone()

        self.x = x
        self.p = p
        self.q = q
        self.beta = beta
        self.sqrt_dt = sqrt_dt
        self.lam = lam
        return

    def find_potential_neighbours(self, k=None, distance_upper_bound=np.inf, workers = -1):
        """
        Uses cKDTree to compute potential nearest-neighbors of each cell

        Parameters
        ----------
            x : array_like
                Position of each cell in 3D space
            k : list of integer or integer
                The list of k-th nearest neighbors to return. If k is an integer it is treated as a list of [1, ... k] (range(1, k+1)). Note that the counting starts from 1.
            distance_upper_bound : nonnegative float, optional
                Return only neighbors within this distance. This is used to prune tree searches, so if you are doing a series of nearest-neighbor queries, it may help to supply the distance to the nearest neighbor of the most recent point. Default: np.inf
            workers: int, optional
                Number of workers to use for parallel processing. If -1 is given, all CPU threads are used. Default: -1.

        Returns
        ----------
            d : array
                distance from each cell to each of its potential neighbors
            idx : array
                index of each cell's potential neighbors
        """
        if k is None:
            k = self.k
        tree = cKDTree(self.x.detach().to("cpu").numpy())
        d, idx = tree.query(self.x.detach().to("cpu").numpy(), k + 1, distance_upper_bound=distance_upper_bound, n_jobs=workers)
        self.d = torch.tensor(d[:, 1:], dtype = self.dtype, device=self.device)
        self.idx = torch.tensor(idx[:, 1:] , dtype = torch.long, device=self.device)

    def find_true_neighbours(self):
        """
        Finds the true neighbors of each cell

        Parameters
        ----------
            d : array
                distance from each cell to each of its potential neighbors
            dx : array
                displacement vector from each cell to each of its potential neighbors

        Returns
        ----------
            z_mask : torch.tensor
        """
        full_n_list = self.x[self.idx]
        self.dx = self.x[:, None, :] - full_n_list
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= self.dx.shape[0]:
                    break
                # ?
                n_dis = torch.sum((self.dx[i0:i1, :, None, :] / 2 - self.dx[i0:i1, None, :, :]) ** 2, dim=3)
                # ??
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

                z_mask = torch.sum(n_dis < (self.d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0
                z_masks.append(z_mask)

                if i1 > self.dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        z_mask = torch.cat(z_masks, dim=0)
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)

        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(self.dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(self.idx, 1, sort_idx)

        m = torch.max(torch.sum(z_mask, dim=1)) + 1

        self.z_mask = z_mask[:, :m]
        self.dx = dx[:, :m]
        self.idx = idx[:, :m]
        self.d = torch.linalg.norm(self.dx, dim = -1)
        return m

    def potential(self, potential):
        """
        Computes the potential energy of the system
        
        Parameters
        ----------
            x : torch.Tensor
                Position of each cell in 3D space
            p : torch.Tensor
                AB polarity vector of each cell
            q : torch.Tensor
                PCP vector of each cell
            idx : array_like
                indices of potential nearest-neighbors of each cell
            d : array_like
                distances from each cell to each of the potential nearest-neighbors specified by idx
            lam : array
                array of weights for the terms that make up the potential
            potential : callable
                function that computes the value of the potential between two cells, i and j
                call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj, **kwargs)
            kwargs : dict
                keyword arguments, passed to the potential function
        
        Returns
        ----------
            V : torch.Tensor
                value of the potential
            m : int
                largest number of true neighbors of any cell
        """
        # Find neighbours
        m = self.find_true_neighbours() # return value = maximum number of true neighbors of any cell
        
        # Normalise dx
        dx = self.dx / torch.linalg.norm(self.dx, dim = -1)[:, :, None]

        # Calculate S
        pi = self.p[:, None, :].expand(self.p.shape[0], self.idx.shape[1], 3)
        pj = self.p[self.idx]
        qi = self.q[:, None, :].expand(self.q.shape[0], self.idx.shape[1], 3)
        qj = self.q[self.idx]

        lam_i = self.lam[:, None, :].expand(self.p.shape[0], self.idx.shape[1], self.lam.shape[1])
        lam_j = self.lam[self.idx]

        Vij = potential(self.x, self.d, dx, lam_i, lam_j, pi, pj, qi, qj)
        V = torch.sum(self.z_mask.float() * Vij)

        return V, int(m)

    def update_k(self, true_neighbour_max, tstep):
        """
        Dynamically adjusts the number of neighbors to look for.

        If very few of the potential neighbors turned out to be true, you can look for fewer potential neighbors next time.
        If very many of the potential neighbors turned out to be true, you should look for more potential neighbors next time.

        Parameters
        ----------
            true_neighbor_max : int
                largest number of true neighbors of any cell found most recently
            tstep : int
                how many time steps of simulation have elapsed

        Returns
        ----------
            k : int
                new max number of potential neighbors to seek
            n_update : int
                controls when to next check for potential neighbors
        """
        k = self.k
        fraction = true_neighbour_max / k
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return n_update

    def gradient_step(self, tstep, potential):
        """
        Move the simulation forward by one time step

        Parameters
        ----------
            dt : float
                Size of the time step
            eta : float
                Strength of the added noise
            lam : torch.Tensor
                weights for the terms of the potential function for each cell.
            beta : torch.Tensor
                for each cell, probability of division per unit time
            p : torch.Tensor
                AB polarity vector of each cell
            q : torch.Tensor
                PCP vector of each cell
            sqrt_dt : float
                square root of dt. To be used for normalizing the size of the noise added per time step
            tstep : int
                how many simulation time steps have elapsed
            x : torch.Tensor
                Position of each cell in 3D space
            potential : callable
                function that computes the value of the potential between two cells, i and j
                call signature (x, d, dx, lam_i, lam_j, pi, pj, qi, qj, **kwargs)
            kwargs : dict
                keyword arguments, to be passed to the potential function

        Returns
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
        """
        # Normalise p, q
        with torch.no_grad():
            self.p /= torch.sqrt(torch.sum(self.p ** 2, dim=1))[:, None]
            self.q /= torch.sqrt(torch.sum(self.q ** 2, dim=1))[:, None]

        # Calculate potential
        V, self.true_neighbour_max = self.potential(potential=potential)

        # Backpropagation
        V.backward()

        # Time-step
        with torch.no_grad():
            self.x += -self.x.grad * self.dt + self.eta * torch.empty(*self.x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            self.p += -self.p.grad * self.dt + self.eta * torch.empty(*self.x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt
            self.q += -self.q.grad * self.dt + self.eta * torch.empty(*self.x.shape, dtype=self.dtype, device=self.device).normal_() * self.sqrt_dt

            if self.callback is not None:
                self.callback(tstep * self.dt, self.x, self.p, self.q, self.lam)

        # Zero gradients
        self.x.grad.zero_()
        self.p.grad.zero_()
        self.q.grad.zero_()

        return

    def simulation(self, potential, division_decider = lambda *args : True):
        """
        Generator to implement the simulation

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
            lam : numpy.ndarray
                weights for the terms of the potential function for each cell.
        """

        tstep = 0
        while True:
            tstep += 1

            # perform cell division, depending on the output of the function division_decider
            # by default, always do cell division (this results in exponential growth of number of cells)
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

            

            self.gradient_step(tstep, potential=potential)

            if tstep % self.yield_every == 0:
                xx = self.x.detach().to("cpu").numpy().copy()
                pp = self.p.detach().to("cpu").numpy().copy()
                qq = self.q.detach().to("cpu").numpy().copy()
                ll = self.lam.detach().to("cpu").numpy().copy()
                yield xx, pp, qq, ll

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
                l0 = self.lam[idx, :]
                b0 = self.beta[idx] * self.beta_decay

                # make a random vector and normalize to get a random direction
                move = torch.empty_like(x0).normal_()
                move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

                # move the cells so that the center of mass of each pair is at the same place as the mother cell was
                x0 = x0 - move/2
                self.x[idx, :] += move/2

                # append new cell data to the system state
                self.x = torch.cat((self.x, x0))
                self.p = torch.cat((self.p, p0))
                self.q = torch.cat((self.q, q0))
                self.lam = torch.cat((self.lam, l0))
                self.beta = torch.cat((self.beta, b0))

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
            l0 = self.lam[idx, :]
            b0 = self.beta[idx] * self.beta_decay

            # make a random vector and normalize to get a random direction
            move = torch.empty_like(x0).normal_()
            move /= torch.sqrt(torch.sum(move**2, dim=1))[:, None]

            # move the cells so that the center of mass of each pair is at the same place as the mother cell was
            x0 = x0 - move/2
            self.x[idx, :] += move/2

            # append new cell data to the system state
            self.x = torch.cat((self.x, x0))
            self.p = torch.cat((self.p, p0))
            self.q = torch.cat((self.q, q0))
            self.lam = torch.cat((self.lam, l0))
            self.beta = torch.cat((self.beta, b0))

        self.x.requires_grad = True
        self.p.requires_grad = True
        self.q.requires_grad = True

        return True




class PolarWNT(Polar):
    def __init__(self, *args, wnt_cells = None, wnt_threshold = 1e-2, wnt_decay = 0, **kwargs):
        self.wnt_cells = wnt_cells
        self.wnt_threshold = wnt_threshold
        self.wnt_decay = wnt_decay
        super().__init__(*args, **kwargs)
        # initialize a G (WNT gradient) vector as zeros
        self.G = torch.zeros_like(self.x)

    def init_simulation(self):
        """
        Checks input dimensions, cleans and converts into torch.Tensor types
        If there are WNT cells, it adds WNT to them in the w tensor

        Parameters
        ----------
            dt : float
                size of time step for simulation
            lam : array
                weights for the terms of the potential function, possibly different for each cell.
            p : array_like
                AB polarity vector of each cell
            q : array_like
                PCP vector of each cell
            w : array_like
                WNT concentration at each cell
            x : array_like
                Position of each cell in 3D space
            beta : array_like
                for each cell, probability of division per unit time
            kwargs : dict
                keyword arguments

        Returns
        ----------
            None
        """
        assert len(self.x) == len(self.p)
        assert len(self.q) == len(self.x)
        assert len(self.beta) == len(self.x)

        sqrt_dt = np.sqrt(self.dt)
        x = torch.tensor(self.x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(self.p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(self.q, requires_grad=True, dtype=self.dtype, device=self.device)
        w = torch.zeros(len(x), requires_grad=False, dtype=self.dtype, device=self.device)
        if self.wnt_cells is not None:
            w[self.wnt_cells] = 1

        beta = torch.tensor(self.beta, dtype=self.dtype, device=self.device)

        lam = torch.tensor(self.lam, dtype=self.dtype, device=self.device)
        # if lam is not given per-cell, return an expanded view
        if len(lam.shape) == 1:
            lam = lam.expand(x.shape[0], lam.shape[0]).clone()

        self.x = x
        self.p = p
        self.q = q
        self.w = w
        self.beta = beta
        self.sqrt_dt = sqrt_dt
        self.lam = lam
        return

    def get_gradient_vectors_better(self):
        self.find_true_neighbours()
        dx = self.dx.clone()
        d = self.d.clone()
        with torch.no_grad():
            # Calculate weights tensors
            w_ii = self.w[:, None].expand(self.w.shape[0], self.idx.shape[1])
            w_ij = torch.zeros_like(w_ii)       # w_ij is zero for all non-weighted cells i
            w_ij = self.w[self.idx] # if cell i has w>threshold, then the i-th row of w_ij will have the w-values of all of its neighbors

            Gij = (-w_ii[:, :, None] + w_ij[:, :, None]) * dx / d[:,:,None]**2
            G = torch.sum(self.z_mask[:, :, None].float() * Gij, dim=1)
        self.G = G
        return G, self.w

    def get_gradient_vectors(self, better = False):
        if better:
            return self.get_gradient_vectors_better()
        # Compute gradient vectors
        self.find_true_neighbours()
        dx = self.dx.clone()
        with torch.no_grad():

            # Count Paneth cells as weighted cells only inside this function to establish gradient
            weighted_cells = torch.where(self.w >= self.wnt_threshold)[0]

            # Calculate weights tensors
            w_ii = self.w[:, None].expand(self.w.shape[0], self.idx.shape[1])
            w_ij = torch.zeros_like(w_ii)       # w_ij is zero for all non-weighted cells i
            w_ij[weighted_cells, :] = self.w[self.idx[weighted_cells]] # if cell i has w>threshold, then the i-th row of w_ij will have the w-values of all of its neighbors

            # Normalize dx compute G
            d_dx = torch.sqrt(torch.sum(dx ** 2, dim=2))
            dx /= d_dx[:, :, None]              # dx[i,j,:] is the unit vector pointing from cell i to cell j. shape is (N, idx.shape[1], 3)
            Gij = (w_ii[:, :, None] + w_ij[:, :, None]) * dx
            G = torch.sum(self.z_mask[:, :, None].float() * Gij, dim=1)

            # Project G vectors so divisions occur in cell-cell-plane and normalize
            G_tilde = -torch.cross(torch.cross(G, self.p), self.p)  # Note the minus!
            # G_tilde = G
            d_G_tilde = torch.sqrt(torch.sum(G_tilde ** 2, dim=1))
            G_tilde[weighted_cells] /= d_G_tilde[weighted_cells, None]
        self.G = G_tilde
        return G_tilde, self.w
    
    def get_gradient_averaging(self):
        # Smoothening the WNT gradient based on true local neighbourhood
        self.find_true_neighbours()
        w_copy = self.w.clone()

        with torch.no_grad():
            weights_true_neighb = self.w[self.idx] * self.z_mask.float()
            w_copy = (torch.sum(weights_true_neighb, dim=1) + w_copy) / (torch.sum(self.z_mask, dim=1) + 1).float()
        
        self.w = w_copy
        # replenish WNT at source cells
        self.w[self.wnt_cells] = 1
        return w_copy

    def potential(self, potential):
        # Find neighbours
        m = self.find_true_neighbours() # return value = maximum number of true neighbors of any cell
        
        # Normalise dx
        dx = self.dx / torch.linalg.norm(self.dx, dim = -1)[:, :, None]

        # Normalise p, q
        with torch.no_grad():
            self.p /= torch.sqrt(torch.sum(self.p ** 2, dim=1))[:, None]
            self.q /= torch.sqrt(torch.sum(self.q ** 2, dim=1))[:, None]

        # Calculate S in the following
        pi = self.p[:, None, :].expand(self.p.shape[0], self.idx.shape[1], 3)
        pj = self.p[self.idx]
        qi = self.q[:, None, :].expand(self.q.shape[0], self.idx.shape[1], 3)
        qj = self.q[self.idx]
        Gi = self.G[:, None, :].expand(self.G.shape[0], self.idx.shape[1], 3)

        lam_i = self.lam[:, None, :].expand(self.p.shape[0], self.idx.shape[1], self.lam.shape[1])
        lam_j = self.lam[self.idx]
        Vij = potential(self.x, self.d, dx, lam_i, lam_j, pi, pj, qi, qj, Gi)
        V = torch.sum(self.z_mask.float() * Vij)

        return V, int(m)

    def simulation(self, potential, division_decider = lambda *args : True, better_WNT_gradient = False):
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

            if tstep % self.yield_every == 0:
                xx = self.x.detach().to("cpu").numpy().copy()
                pp = self.p.detach().to("cpu").numpy().copy()
                qq = self.q.detach().to("cpu").numpy().copy()
                ww = self.w.detach().to("cpu").numpy().copy()
                ll = self.lam.detach().to("cpu").numpy().copy()
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

                # append new cell data to the system state
                self.x = torch.cat((self.x, x0))
                self.p = torch.cat((self.p, p0))
                self.q = torch.cat((self.q, q0))
                self.w = torch.cat((self.w, w0))
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

            # append new cell data to the system state
            self.x = torch.cat((self.x, x0))
            self.p = torch.cat((self.p, p0))
            self.q = torch.cat((self.q, q0))
            self.w = torch.cat((self.w, w0))
            self.lam = torch.cat((self.lam, l0))
            self.beta = torch.cat((self.beta, b0))

        # replenish WNT at source cell(s)
        self.w[self.wnt_cells] = 1

        self.x.requires_grad = True
        self.p.requires_grad = True
        self.q.requires_grad = True

        return True
