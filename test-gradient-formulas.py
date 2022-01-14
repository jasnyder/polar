import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

n = 30
G = nx.grid_2d_graph(n, n)

def fun(x, y, n = 30, sigma = 10):
    return np.exp(((x-0.5*n)**2+(y-0.5*n)**2)/sigma**2)

for x, y in G.nodes:
    nx

print(list(G.nodes)[:10])