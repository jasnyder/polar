import pickle
import numpy as np
import pandas as pd
import plotly.express as px

with open('data/dump/ligand_tree_fail.pkl','rb') as fobj:
    mdict = pickle.load(fobj)
    ligand = mdict['ligand']

df = pd.DataFrame(ligand.numpy(), columns = ['x1', 'x2', 'x3'])

fig = px.scatter_3d(df, x='x1', y='x2', z='x3')
fig.write_html('test_figs/ligand_tree_fail.html')
