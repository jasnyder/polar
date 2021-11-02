import numpy as np
import pickle

def init_grid(n, m, d = 2):
    """
    Initializes a square grid of cells at a distance of d cell radii (default = 2)

    AB polarity is initialized as 'up', and PCP is a random direction in the plane
    """
    xx = np.arange(n) - n/2 + 1/2
    yy = np.arange(m) - m/2 + 1/2
    idx = 0
    x = np.zeros((n*m, 3))
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    Q = np.random.randn(3)
    Q = Q/np.sqrt((Q**2).sum())
    for X in xx:
        for Y in yy:
            x[idx, :] = [X*d, Y*d, 0]
            p[idx, :] = [0, 0, 1]
            q[idx, :] = Q
            idx += 1
    return x, p, q

def init_random_system(n):
    """
    Initialize a random system with n cells. Position is normally distributed in each dimension, and each component of p and of q is uniformly distributed between -1 and 1
    
    Parameters
    ---------
    n : int
        Number of cells

    Returns
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell
    """
    x = np.random.randn(n, 3)
    p = 2 * np.random.rand(n, 3) - 1
    q = 2 * np.random.rand(n, 3) - 1

    return x, p, q

def init_sphere(n, R = None):
    """
    Initialize a system with n points arranged in a sphere, with AB polarity pointing outward and PCP oriented randomly.

    To get x to lie on a sphere, sample x randomly and then normalize it.
    To get AB polarity to point outward, set it proportional to x

    Parameters
    ----------
    n : int
        Number of cells
    R : float, optional
        Radius of the sphere on which to place cells. If not specified, R is set so that the sphere's surface area is one per cell

    Returns
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell
    """

    if R is None:
        # set R so that on average each cell takes up unit area on the surface of the sphere
        R = np.sqrt(n/(4*np.pi))
    x = np.random.randn(n, 3)
    x = R*(x.T/np.linalg.norm(x, axis = 1)).T

    p = x/R
    q = 2*np.random.rand(n, 3) - 1

    return x, p, q

def init_tube(n, R = None, L = None, comb_direction = None):
    """
    Initializes a tube of radius R and length L.
    If either L or R is not specified, it is set so that each cell has a unit area on the tube.
    If neither L nor R is specified, L is set to 10*R

    AB polarity points out of the tube.
    PCP is oriented randomly.

    Parameters
    ----------
    n : int
        Number of cells
    R : float, optional
        Radius of the tube. Default: set so that each cell has a unit area on average
    L : float, optional
        Length of the tube. Default: 10*R
    comb_direction : str or None, optional
        direction in which to align q vectors, either along or around the tube.
        If None (default), leave q as random

    Returns
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell
    """
    if R is None and L is None:
        R = np.sqrt(n/(20*np.pi))
        L = 10*R
    elif L is None:
        L = n/(2*np.pi*R)
    elif R is None:
        R = n/(2*np.pi*L)

    # sample angle around tube and distance along the tube uniformly at random
    phi = 2*np.pi*np.random.rand(n)
    l = L*np.random.rand(n) - L/2
    # combine into 3D coordinates
    x = np.array([R*np.cos(phi), R*np.sin(phi), l]).T

    # make AB polarity point straight out of the tube and normalize to have length 1
    p = x.copy() / R
    p[:,2] = 0

    # initialize PCP at random
    q = 2*np.random.rand(*x.shape) - 1

    if comb_direction is not None:
        return comb_tube(x, p, q, direction = comb_direction)
    else:
        return x, p, q

def init_tube_grid(n, R = None, L = None, comb_direction = None):
    """
    Initializes a system with cells arranged on a tube, placed deterministically to be evenly spaced (i.e. on a grid)
    I'm doing this lazily so it may not end up with exactly n cells. Sorry.

    Parameters
    ----------
    n : int
        Number of cells
    R : float, optional
        Radius of the tube. Default: set so that each cell has a unit area on average
    L : float, optional
        Length of the tube. Default: 10*R
    comb_direction : str or None, optional
        direction in which to align q vectors, either along or around the tube.
        If None (default), leave q as random

    Returns
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell
    """
    if R is None and L is None:
        R = np.sqrt(n/(20*np.pi))
        L = 10*R
    elif L is None:
        L = n/(2*np.pi*R)
    elif R is None:
        R = n/(2*np.pi*L)

    # generate distance along the tube and angle around the tube evenly spaced
    l = 0.5 + np.arange(-L/2, L/2)
    phi = 2*np.pi*np.linspace(0, 1, int(n/L), endpoint = False)

    # combine into 3D coordinates
    x = np.empty((len(l)*len(phi), 3), dtype = float)
    idx = 0
    dphi = np.pi / len(phi)
    for ph in phi:
        for j, ell in enumerate(l):
            x[idx, :] = np.array([R*np.cos(ph + j*dphi), R*np.sin(ph + j*dphi), ell])
            idx+=1

    # make AB polarity point straight out of the tube and normalize to have length 1
    p = x.copy() / R
    p[:,2] = 0

    # initialize PCP, either around the tube or at random
    q = 2*np.random.rand(*x.shape) - 1

    if comb_direction is not None:
        return comb_tube(x, p, q, direction = comb_direction)
    else:
        return x, p, q

def comb_tube(x, p, q, direction = 'around'):
    """
    align the PCP vectors (q) to point around the tube on which cells are positioned (x).

    This function assumes that:
        x.shape = (n, 3) and each x[i,:] lies on a tube centered on the z-axis
        p[:, :2] = x[:, :2] and p[:, 2] = 0

    Parameters:
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell

    Returns
    ----------
    x : np.ndarray
        Position of each cell in 3D space
    p : np.ndarray
        AB polarity vector of each cell
    q : np.ndarray
        PCP vector of each cell
    """
    if direction == 'around':
        q = np.vstack((p[:,1], -p[:,0], p[:,2])).T
    elif direction == 'along':
        q = np.zeros_like(q)
        q[:, 2] = 1
    else:
        raise NotImplementedError('direction must be either \'around\' or \'along\'')
    return x, p, q

def comb_sphere(x, p, q):
    q = np.vstack((p[:,1], -p[:,0], np.zeros_like(p[:,2]))).T
    return x, p, q

def load_cached(fname):
    with open(fname, 'rb') as fobj:
        x, p, q = pickle.load(fobj)
    return x, p, q