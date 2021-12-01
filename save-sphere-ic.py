import numpy as np
import pickle

fname_in = 'data/ic/relaxing-sphere.pkl'


with open(fname_in, 'rb') as fobj:
    data, kwargs = pickle.load(fobj)

x = data[-1][0]
p = data[-1][1]
q = data[-1][2]

n = len(x)

fname_out = f'data/ic/relaxed-sphere-n-{n}.pkl'
with open(fname_out, 'wb') as fobj:
    pickle.dump([x, p, q], fobj)