import numpy as np


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

def init_tube(n, R = None, L = None):
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

    # make PCP random
    q = 2*np.random.rand(n, 3) - 1

    return x, p, q

def init_tube_grid(n, R = None, L = None):
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
    phi = 2*np.pi*np.linspace(0, 1, int(n/L))

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

    # make PCP random
    q = 2*np.random.rand(*x.shape) - 1

    return x, p, q
