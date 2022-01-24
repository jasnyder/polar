"""
this file is for me to play around with visibility computation

The purpose is to potentially use this as a proxy for diffusion of FGF in the mesenchyme. The idea is that the amount (weighted by receptor level) of epithelial surface that's "visible to" a given cell should determine how much FGF is able to reach that cell.

To do this I'm going to play around with the method proposed by Katz that involves inverting the point cloud across a sphere centered at the "camera" (i.e. focal cell) and finding the convex hull of that point.

I'm concerned with what this algorithm will do when the camera is directly on the cell surface (i.e. it is one of the cells). So let's try this out with the sphere IC
"""
import plotly.express as px
import numpy as np
import scipy as sp
from scipy import spatial
import matplotlib.pyplot as plt
import time
import itertools
import os
from polarcore import PolarWNT
from initsystems import init_random_system, init_tube, init_tube_grid
from plot.plotcore import load, build_df_wnt, select
import plotly.express as px
import potentials_wnt
import pickle

n = 1000
with open(f'data/ic/relaxed-sphere-n-{n}.pkl', 'rb') as fobj:
    x, p, q = pickle.load(fobj)

# pick a focal cell
focal = 0
C = x[focal]

# invert point cloud through a sphere of radius R


def invert(x, C, R=None, R_pow=1):
    xshifted = x-C
    xnorm = np.linalg.norm(xshifted, axis=1)
    if R is None:
        R = xnorm.max() * 10**R_pow
    out = xshifted + 2*(R-xnorm[:, None])*(xshifted/xnorm[:, None])
    # if any points were coincident with C, set their image equal to zero
    out[xnorm == 0] = 0
    return out


def visible_indices(x, C, R=None, R_pow=1):
    phat = invert(x, C, R=R, R_pow=R_pow)
    hull = sp.spatial.ConvexHull(phat)
    return hull.vertices


# dither the camera point to be outside of the sphere by a bit
phat = invert(x, C+200*p[focal])
hull = sp.spatial.ConvexHull(phat)
len(hull.vertices)


"""
Try a 2d example because visualization is easier
"""
N = 500
theta = np.linspace(-np.pi, np.pi, num=N)
x = np.array([np.cos(theta), np.sin(theta)]).T

focal = 0
dither = 0.0
R_pow = 0.5
offset = dither * x[focal]
C = x[focal] + offset
phat = invert(x, C, R_pow=R_pow)
hull = sp.spatial.ConvexHull(phat)
v = hull.vertices
plt.scatter(*x.T)
plt.scatter(x[v, 0], x[v, 1], color='red')
plt.scatter(*C, color='green')

"""
Try to load a configuration from a previous run
"""
fname = 'data/sphere-wnt-reweighting-n-1000-n_wnt-5-07Jan2022-10-43-25.pkl'
data, kwargs, fname = load(fname)
df, kwargs = build_df_wnt(data, kwargs=kwargs)
df = select(df, -1)
df.index = df.i.astype(int)

def plot(df, v, focal):
    vcol = np.zeros_like(df.x1)
    vcol[v] = 1
    vcol[focal] = 2
    df['v'] = vcol

    range_x = (df['x1'].min(), df['x1'].max())
    range_y = (df['x2'].min(), df['x2'].max())
    range_z = (df['x3'].min(), df['x3'].max())
    fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                        color='v', range_x=range_x, range_y=range_y, range_z=range_z)
    return fig

def save(fig, fname):
    fig.write_html(fname.replace('data', 'test_figs').replace('.pkl', '.html'), include_plotlyjs='directory',
                   full_html=False)

x = np.array([df.x1, df.x2, df.x3]).T
p = np.array([df.p1, df.p2, df.p3]).T
focal = np.random.randint(len(x))
dither = 2.0
C = x[focal] + dither*p[focal]


R_pow = 0.1
phat = invert(x, C, R_pow=R_pow)
hull = sp.spatial.ConvexHull(phat)
v = hull.vertices
fig = plot(df, v, focal)
save(fig, fname)