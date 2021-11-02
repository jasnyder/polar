import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pickle

fname = input('Enter data filename: ')#'data/test1.pkl'

print('Loading data from file '+fname)
has_kwargs = False
with open(fname, 'rb') as f:
    try:
        data, kwargs = pickle.load(f)
        has_kwargs = True
    except ValueError:
        data = pickle.load(f)  # contains x, p, q, lam
# data[t][0] == x, x[i, k] = position of particle i in dimension k
# data[t][1] == p, p[i, k] = AB polarity of particle i in dimension k
# data[t][2] == q, q[i, k] = PCP of particle i in dimension k

# create dataframe
row_chunks = list()
for t, dat in enumerate(data):
    if has_kwargs:
        T = kwargs['dt'] * kwargs['yield_every'] * t
    else:
        T = t
    n = dat[0].shape[0]
    row_chunks.append(np.hstack([np.ones((n,1)) * T, np.arange(n)[:,np.newaxis], dat[0], dat[1], dat[2]]))

df = pd.DataFrame(np.vstack(row_chunks), columns = ['t', 'i', 'x1', 'x2', 'x3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3'])

tmax = df['t'].max()
fig = go.Figure(data = [go.Cone(
        x=df.loc[df['t']==tmax]['x1'],
        y=df.loc[df['t']==tmax]['x2'],
        z=df.loc[df['t']==tmax]['x3'],
        u=df.loc[df['t']==tmax]['q1'],
        v=df.loc[df['t']==tmax]['q2'],
        w=df.loc[df['t']==tmax]['q3'],
        sizemode = 'absolute',
        sizeref = 2
    )])

def fun(scene):
    scene.aspectmode = 'data'
    return
fig.for_each_scene(fun)

fig.write_html('animations/'+fname.split('/')[1].split('.')[0]+'_vectorfield.html', include_plotlyjs = 'directory', full_html = False, animation_opts = {'frame':{'duration':100}})
