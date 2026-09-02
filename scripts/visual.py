# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:38:57 2020

@author: JiaLi
"""

import matplotlib.pyplot as plt
import networkx as nx

def plot_adjMatx(adj_matx,cmap = 'Greys',filePath = 'False'):
    """
    Plot adjacency matrix.
    """
    plt.imshow(adj_matx, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

def plot_network(vertices_coord, adj_matrix):
    """
    Plot network in the topogaphical view.
    """
    G = nx.from_numpy_matrix(adj_matrix.T, create_using=nx.DiGraph)
    pos = dict(enumerate(vertices_coord))
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    
    nx.draw_networkx(G, pos, node_size=20, node_color='b', edge_color=weights,
                     width=1.0, edge_cmap=plt.cm.Greys, with_labels=False)
