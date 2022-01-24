import plotly.graph_objects as go
import numpy as np
import torch

from plot.plotcore import load, build_df_wnt, select

from polarcore import PolarWNT


def compute_WNT_gradient(df, kwargs, project = False):
    """
    takes a dataframe of a single timeslice and builds a PolarWNT object
    the PolarWNT object knows how to find potential/true neighbors, and compute WNT gradients
    """
    x = np.vstack([df['x1'], df['x2'], df['x3']]).T
    p = np.vstack([df['p1'], df['p2'], df['p3']]).T
    q = np.vstack([df['q1'], df['q2'], df['q3']]).T
    w = torch.tensor(np.array(df['w']))
    lam = np.array(kwargs['lam'].cpu())[:len(x)]
    beta = np.array(kwargs['beta'].cpu())[:len(x)]
    wnt_cells = kwargs['wnt_cells']
    wnt_threshold = kwargs['wnt_threshold']
    wnt_decay = kwargs['wnt_decay']
    eta = kwargs['eta']
    yield_every = kwargs['yield_every']
    beta_decay = kwargs['beta_decay']
    sim = PolarWNT(x, p, q, lam, beta, wnt_cells=wnt_cells, wnt_threshold = wnt_threshold, wnt_decay=wnt_decay, eta=eta, yield_every=yield_every,
               device="cpu", init_k=50, beta_decay=beta_decay, divide_single=True, dtype = torch.float64)
    sim.w = w
    sim.find_potential_neighbours()
    Gtilde, w = sim.get_gradient_vectors_unnormalized(project = project)
    df['G1'], df['G2'], df['G3'] = Gtilde.T
    return df, kwargs


def plot(df):
    fig = go.Figure(data=[go.Cone(
        x=df['x1'],
        y=df['x2'],
        z=df['x3'],
        u=df['G1'],
        v=df['G2'],
        w=df['G3'],
        sizemode='absolute',
        sizeref=2,
    )])

    def fun(scene):
        scene.aspectmode = 'data'
        return
    fig.for_each_scene(fun)
    return fig


def save(fig, fname, wnt_gradient_version = ''):
    fig.write_html(fname.replace('data', 'animations').replace('.pkl', f'_{wnt_gradient_version}_WNT_gradient.html'),
                   include_plotlyjs='directory', full_html=False, animation_opts={'frame': {'duration': 100}})


if __name__ == "__main__":
    fname = input('Enter data filename: ')  # 'data/test1.pkl'
    T_plot = int(input('timestep to plot: ') or -1)
    wnt_gradient_version = input('project perp to AB? (default: False) ') or False
    better_wnt_gradient = (wnt_gradient_version=='new')
    data, kwargs, fname = load(fname)
    df, kwargs = build_df_wnt(data, kwargs)
    df_t = select(df, T_plot, kwargs)
    df, kwargs = compute_WNT_gradient(df_t, kwargs, project = wnt_gradient_version)
    fig = plot(df)
    save(fig, fname, wnt_gradient_version=wnt_gradient_version)
