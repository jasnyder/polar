import numpy as np
import pandas as pd
import os
import pickle

def load(fname):
    if fname == "most recent":
        max_mtime = 0
        for dirname,subdirs,files in os.walk("./data/"):
            for file in files:
                full_path = os.path.join(dirname, file)
                mtime = os.stat(full_path).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    max_file = full_path
        fname = max_file
    print('Loading data from file '+fname)
    with open(fname, 'rb') as f:
        try:
            data, kwargs = pickle.load(f)
        except ValueError:
            data = pickle.load(f)  # contains x, p, q, lam
            kwargs = None
    # data[t][0] == x, x[i, k] = position of particle i in dimension k
    # data[t][1] == p, p[i, k] = AB polarity of particle i in dimension k
    # data[t][2] == q, q[i, k] = PCP of particle i in dimension k
    return data, kwargs, fname


def build_df(data, kwargs=None):
    # create dataframe
    row_chunks = list()
    for t, dat in enumerate(data):
        if kwargs is not None:
            T = kwargs['dt'] * kwargs['yield_every'] * t
        else:
            T = t
        n = dat[0].shape[0]
        row_chunks.append(np.hstack(
            [np.ones((n, 1)) * T, np.arange(n)[:, np.newaxis], dat[0], dat[1], dat[2]]))

    df = pd.DataFrame(np.vstack(row_chunks), columns=[
                      't', 'i', 'x1', 'x2', 'x3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3'])
    return df, kwargs

def build_df_wnt(data, kwargs = None):
    row_chunks = list()
    for t, dat in enumerate(data):
        if kwargs is not None:
            T = kwargs['dt'] * kwargs['yield_every'] * t
        else:
            T = t
        n = dat[0].shape[0]
        row_chunks.append(np.hstack([np.ones((n,1)) * T, np.arange(n)[:,np.newaxis], dat[0], dat[1], dat[2], dat[3][:,None]]))

    df = pd.DataFrame(np.vstack(row_chunks), columns = ['t', 'i', 'x1', 'x2', 'x3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'w'])
    return df, kwargs


def select(df, T_plot, kwargs=None):
    if kwargs is None:
        ye = 1
        dt = 1
    else:
        ye = kwargs['yield_every']
        dt = kwargs['dt']
    if T_plot == -1:
        tt = df['t'].max()
    else:
        tt = T_plot * ye * dt
    return df.loc[df['t'] == tt]
