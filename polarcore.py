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
                 device='cuda', dtype=torch.float, init_k=100, callback=None, wnt_cells = None, wnt_range = None):
        self.device = device
        self.dtype = dtype

        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None
        self.callback = callback

        self.x = None
        self.p = None
        self.q = None
        self.beta = None
        self.dt = dt
        self.sqrt_dt = None
        self.lam = None

        self.init_simulation(x, p, q, lam, beta)
        self.eta = eta
        self.yield_every = yield_every
        self.beta_decay = beta_decay
        self.do_nothing_threshold = do_nothing_threshold
        self.divide_single = divide_single
        self.wnt_cells = wnt_cells
        self.wnt_range = wnt_range

    def init_simulation(self, x, p, q, lam, beta):
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
        assert len(x) == len(p)
        assert len(q) == len(x)
        assert len(beta) == len(x)

        sqrt_dt = np.sqrt(self.dt)
        x = torch.tensor(x, requires_grad=True, dtype=self.dtype, device=self.device)
        p = torch.tensor(p, requires_grad=True, dtype=self.dtype, device=self.device)
        q = torch.tensor(q, requires_grad=True, dtype=self.dtype, device=self.device)

        beta = torch.tensor(beta, dtype=self.dtype, device=self.device)

        lam = torch.tensor(lam, dtype=self.dtype, device=self.device)
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
        return d[:, 1:], idx[:, 1:]

    def find_true_neighbours(self, d, dx):
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
        with torch.no_grad():
            z_masks = []
            i0 = 0
            batch_size = 250
            i1 = batch_size
            while True:
                if i0 >= dx.shape[0]:
                    break
                # ?
                n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
                # ??
                n_dis += 1000 * torch.eye(n_dis.shape[1], device=self.device, dtype=self.dtype)[None, :, :]

                z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0
                z_masks.append(z_mask)

                if i1 > dx.shape[0]:
                    break
                i0 = i1
                i1 += batch_size
        z_mask = torch.cat(z_masks, dim=0)
        return z_mask

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

        full_n_list = self.x[self.idx]
        dx = self.x[:, None, :] - full_n_list
        z_mask = self.find_true_neighbours(self.d, dx)

        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask.int(), dim=1, descending=True)

        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx = torch.gather(self.idx, 1, sort_idx)

        
        m = torch.max(torch.sum(z_mask, dim=1)) + 1

        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx[:, :m]

        # Normalise dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))
        dx = dx / d[:, :, None]

        # Calculate S
        pi = self.p[:, None, :].expand(self.p.shape[0], idx.shape[1], 3)
        pj = self.p[idx]
        qi = self.q[:, None, :].expand(self.q.shape[0], idx.shape[1], 3)
        qj = self.q[idx]

        lam_i = self.lam[:, None, :].expand(self.p.shape[0], idx.shape[1], self.lam.shape[1])
        lam_j = self.lam[idx]

        if self.wnt_cells is not None:
            Vij = potential(self.x, d, dx, lam_i, lam_j, pi, pj, qi, qj, self.wnt_cells, self.wnt_range)
        else:
            Vij = potential(self.x, d, dx, lam_i, lam_j, pi, pj, qi, qj)
        V = torch.sum(z_mask.float() * Vij)

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
                d, idx = self.find_potential_neighbours()
                self.idx = torch.tensor(idx, dtype=torch.long, device=self.device)
                self.d = torch.tensor(d, dtype=self.dtype, device=self.device)
            

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
