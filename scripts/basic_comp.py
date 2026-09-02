# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:24:35 2020

@author: JiaLi
"""

import numpy as np
from scipy import linalg
from scipy.spatial.distance import squareform, pdist

def generate_nodes_coord(n_nodes):
    """
    Generate spatial embedding for nodes. Nodes are randomly positioned on a unit disk.
    args:
        n_nodes: number of nodes
    returns:
        nodes_coord: positions of nodes on the unit dist
    """
    # generate random angular coordinates
    phi = np.random.random(size=n_nodes) * 2 * np.pi
    
    # generate random radial coordinates
    r2 = np.random.random(size=n_nodes)
    
    # transfer into Cartesian coordinates
    x = np.sqrt(r2)*np.cos(phi)
    y = np.sqrt(r2)*np.sin(phi)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    nodes_coord = np.concatenate((x,y),axis=1)
    
    return nodes_coord

def generate_rand_adj(n_nodes, edges, weight_distribution,**kwargs):
    """
    Generate an adjacency matrix of a random directed weighted network.
    Adapt from Rentzeperis et al., 2022.
    args:
        n_nodes: number of nodes
        edges: number of edges
        **kwargs
                weight_distribution:
                    'binary', 'normal' or 'lognormal'
                mu:
                    the mu parameter, is not valid for binary
                sig:
                    the sig parameter, is not valid for binary
    returns:
        rand_adj_matrix: adjacency matrix
    """
    # settings of weight distribution
    for (key, value,) in kwargs.items():
        if key == "weight_distribution":
            weight_distribution = value
        elif key == "mu":
            mu = value
        elif key == "sig":
            sig = value

    # set constant
    EPSILON = 0.05

    # set the max number of network edges
    max_connections = int(n_nodes * (n_nodes - 1))
    
    if edges > max_connections or edges < 0:
        print("Edge number out of range")
        return -1
        
    print("Generating random adjacency matrix ...")

    # sample weights from a lognormal distribution
    if weight_distribution == "lognormal":
        rand_weights = np.random.lognormal(mean=mu, sigma=sig, size=edges,)

    # ... from a normal distribution
    elif weight_distribution == "normal":
        rand_weights = np.random.normal(loc=mu, scale=sig, size=edges,)
        ind = np.where(rand_weights < 0)
        rand_weights[ind] = EPSILON

    # ... from a binary distribution
    elif weight_distribution == "binary":
        rand_weights = np.ones(edges)

    # Normalize weights such that their sum equals the number of edges
    if (weight_distribution == "normal") | (weight_distribution == "lognormal"):
        norm_factor = edges / np.sum(rand_weights)
        norm_rand_weights = rand_weights * norm_factor
    else:
        norm_rand_weights = rand_weights

    # Get the indices of 1s of a matrix the same size as A with 1s everywhere except in the diagonal
    Aones = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    ind = np.where(Aones)

    # Pick a random sample of those indices (# edges)
    rand_max_con = np.random.permutation(max_connections)
    rand_edges_ind = (ind[0][rand_max_con[:edges]],ind[1][rand_max_con[:edges]],)

    # build the adjacency matrix w/ those indices
    adj_matx = np.zeros((n_nodes, n_nodes))
    adj_matx[rand_edges_ind] = norm_rand_weights

    return adj_matx

def initial_directed_network(n_nodes, edges, **kwargs):
    """
    Generate an random directed weighted network with spatial embedding in a unit disk.
    args:
        n_nodes: number of nodes
        edges: number of edges
        **kwargs:
                weight_distribution:
                    'binary', 'normal' or 'lognormal'
                mu:
                    the mu parameter, is not valid for binary
                sig:
                    the sig parameter, is not valid for binary
    returns:
        nodes_coord: positions of nodes in the unit dist
        rand_adj_matrix: adjacency matrix
    """
    # generate adjacency matrix
    rand_adj_matrix = generate_rand_adj(n_nodes, edges, **kwargs)
    # generate node positions
    nodes_coord = generate_nodes_coord(n_nodes)
    
    return nodes_coord, rand_adj_matrix

def compute_distance_matrix(vertices_coord):
    """
    Compute the spatial distances between nodes.
    args:
        vertices_coord: Cartesian coordinates of nodes
    returns:
        D: Euclidean distance matrix whose entry d_ij is the spatial distance between node i and node j.
    """
    D = squareform(pdist(vertices_coord))
    return D

def compute_wave_matrix(vertices_coord, vector_field, D, flag):
    """
    Compute the alignment matrix W.
    The entry w_ij=1-cos(angle between the potential edges j->i and the vector field), which is proportional to the angle.
    cos(j->i, v_i) = <j->i,v_i>/(||j->i||*||v_i||)
    cos(j->i, v_j) = <j->i,v_j>/(||j->i||*||v_j||)
    <j->i,v_j> is the inner product of vector j->i and the vector field v_j
    ||j->i|| is the norm (Euclidean distance) of edge j->i, ||v_j|| is the norm of the vector field at node j
    
    args:
        vertices_coord: Cartesian coordinates of nodes
        vector field: an #nodes*2 array, containing the local vector field for each node
        D: distance matrix
        flag: indicate rewiring in-link or out-link
    returns:
        W_in: angle matrix whose entry w_ij is 1-cos(angle between edges j->i and the vector potential at node i)
        W_out: angle matrix whose entry w_ij is 1-cos(angle between edges j->i and the vector potential at node j)
    """
    # denom_cos_theta_ij = ||j->i||*||v_i|| = ||i->j||*||v_i||
    denom_cos_theta = D*(linalg.norm(vector_field,axis=1)[:,np.newaxis])
    
    # make denom_cos_theta_ii = 1 to avoid NaN
    np.fill_diagonal(denom_cos_theta,1)
        
    # if out-link is rewired
    if flag == 'out':
        # nom_cos_theta_out_ij = <i->j,v_i>
        nom_cos_theta_out = np.sum(-(vertices_coord[:,np.newaxis,:]-vertices_coord)*vector_field[:,np.newaxis],axis=2)
        # make nom_cos_theta_out_ii = 1 so that W_out_ii = 0
        np.fill_diagonal(nom_cos_theta_out,1)
        
        W_out = (1-nom_cos_theta_out/denom_cos_theta).T
        return W_out
    
    # if in-link is rewired
    elif flag == 'in':
        # nom_cos_theta_in_ij = <j->i,v_i>
        nom_cos_theta_in = np.sum((vertices_coord[:,np.newaxis,:]-vertices_coord)*vector_field[:,np.newaxis],axis=2)
        # make nom_cos_theta_in_ii = 1 so that W_in_ii = 0
        np.fill_diagonal(nom_cos_theta_in,1)
        
        W_in = (1-nom_cos_theta_in/denom_cos_theta)
        return W_in

def compute_consensus_kernel(weight_matx, tau):
    """
    Compute the consensus kernel.
    args:
        weight_matx: adjacency matrix
        tau: time constant
    returns:
        kernel: consensus kernel
    """
    # estimate the in degree Laplacian
    Din = np.diag(np.sum(weight_matx, axis=1))
    Lin = Din - weight_matx

    # calculate the consensus kernel
    kernel = linalg.expm(-tau * Lin)
    kernel[kernel<(-1e-15)] = 0
    return kernel

def compute_advection_kernel(weight_matx, tau):
    """
    Compute the advection kernel.
    args:
        weight_matx: adjacency matrix
        tau: time constant
    returns:
        kernel: advection kernel
    """
    # estimate the out degree Laplacian
    Dout = np.diag(np.sum(weight_matx, axis=0))
    Lout = Dout - weight_matx

    # calculate the advection kernel
    kernel = linalg.expm(-tau * Lout)
    kernel[kernel<(-1e-15)] = 0
    return kernel

def choose_rewire_vertex(A, nodes_rew, flag):
    """
    Generate a node v whose link will be rewired, v's neighbors, and the rest nodes.
    args:
        A: adjacency matrix
        nodes_rew: rewirable nodes, i.e., with its in-/out-degree >0 and <n-1
        flag: indicate rewiring in-link or out-link
    returns:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
    """
    # randomly select node v from the rewirable nodes
    v = np.random.choice(nodes_rew)
    all_vertices_ind = np.arange(A.shape[0])
    no_v_vertices_ind = np.delete(all_vertices_ind, v)
    
    if flag=='in':
        U_cp_v = no_v_vertices_ind[np.where(A[v, no_v_vertices_ind]>0)[0]]
        U_nc_v = no_v_vertices_ind[np.where(A[v, no_v_vertices_ind]==0)[0]]
    elif flag=='out':
        U_cp_v = no_v_vertices_ind[np.where(A[no_v_vertices_ind, v]>0)[0]]
        U_nc_v = no_v_vertices_ind[np.where(A[no_v_vertices_ind,v]==0)[0]]
    return v, U_cp_v, U_nc_v

def rand_rewiring(U_cp_v, U_nc_v):
    """
    Perform random rewiring
    args:
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
    """
    j_minus = np.random.choice(U_cp_v)
    j_add = np.random.choice(U_nc_v)
    return j_add, j_minus

def dist_rewiring(v, U_cp_v, U_nc_v, D):
    """
    Perform distance rewiring principle
    args:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
        D: spatial distance matrix
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
    """
    j_minus = U_cp_v[np.argmax(D[U_cp_v,v])]
    j_add = U_nc_v[np.argmin(D[U_nc_v,v])]
    return j_add, j_minus
    
def wave_rewiring(v, U_cp_v, U_nc_v, W, flag):
    """
    Perform wave rewiring principle
    args:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
        W: alignment matrix
        flag: indicate rewiring in-link or out-link
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
    """
    if flag == 'out':
        j_minus = U_cp_v[np.argmax(W[U_cp_v,v])]
        j_add = U_nc_v[np.argmin(W[U_nc_v,v])]
    elif flag == 'in':
        j_minus = U_cp_v[np.argmax(W[v,U_cp_v])]
        j_add = U_nc_v[np.argmin(W[v,U_nc_v])]
    return j_add, j_minus

def adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag):
    """
    Perform adaptive rewiring principle
    args:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
        A: adjacency matrix
        tau: time constant
        flag: indicate rewiring in-link or out-link
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
        tie_cp_flag: indicate tie of minimums in kernel values of U_cp_v
        tie_nc_flag: indicate tie of maximums in kernel values of U_nc_v
    """  
    if flag == 'out':
        L = compute_advection_kernel(A, tau)
        # kernel values of nodes in U_cp_v and U_nc_v
        states_cp = L[U_cp_v,v]
        states_nc = L[U_nc_v,v]
    elif flag == 'in':
        L = compute_consensus_kernel(A, tau) 
        # kernel values of nodes in U_cp_v and U_nc_v
        states_cp = L[v,U_cp_v]
        states_nc = L[v,U_nc_v]
    
    # check there are more than one nodes in U_cp_v with minimum kernel value
    # if true, randomly choose among them
    state_min = np.min(states_cp)
    tie_cp = np.where(states_cp==state_min)[0]
    if len(tie_cp)>1:
        ind_minus = np.random.choice(tie_cp)
    else:
        ind_minus = tie_cp[0]
    # flag of tie
    tie_cp_flag = int(len(tie_cp)>1)

    # check there are more than one nodes in U_nc_v with maximum kernel value
    # if true, randomly choose among them
    state_max = np.max(states_nc)
    tie_nc = np.where(states_nc==state_max)[0]
    if len(tie_nc)>1:
        ind_add = np.random.choice(tie_nc)
    else:
        ind_add = tie_nc[0]
    # flag of tie
    tie_nc_flag = int(len(tie_nc)>1)

    j_minus = U_cp_v[ind_minus]
    j_add = U_nc_v[ind_add]

    return j_add, j_minus, tie_cp_flag, tie_nc_flag

################## stochastic version ####################
def stoch_dist_rewiring(v, U_cp_v, U_nc_v, D):
    """
    Perform distance rewiring principle
    args:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
        D: spatial distance matrix
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
    """
    states_cp = D[U_cp_v,v]
    states_nc = D[U_nc_v,v]
    
    # choose the node that edge detached from with probability proportional to 1/states_cp
    states_temp = 1/states_cp
    prob_cp = states_temp/np.sum(states_temp)
    j_minus = np.random.choice(U_cp_v,size=1,replace=False,p=prob_cp)

    # choose the node that edge reconnected to with probability proportional to states_nc
    prob_nc = states_nc/np.sum(states_nc)
    j_add = np.random.choice(U_nc_v,size=1,replace=False,p=prob_nc)
    return j_add, j_minus
    
def stoch_adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag):
    """
    Perform stochastic version of adaptive rewiring principle. 
    Select the nodes detached from and reconnected to with probability proportional to kernel values.
    args:
        v: node whose link will be rewired
        U_cp_v: v's neighborhood
        U_nc_v: nodes that are not connected to v
        A: adjacency matrix
        tau: time constant
        flag: indicate rewiring in-link or out-link
    returns:
        j_minus: node from which the rewired edge detach
        j_add: node to which the rewired edge reconnect
        tie_cp_flag: indicate tie of minimums in kernel values of U_cp_v
        tie_nc_flag: indicate tie of maximums in kernel values of U_nc_v
    """ 
    if flag == 'out':
        L = compute_advection_kernel(A, tau)
        # kernel values of nodes in U_cp_v and U_nc_v
        states_cp = L[U_cp_v,v]
        states_nc = L[U_nc_v,v]
    elif flag == 'in':
        L = compute_consensus_kernel(A, tau) 
        # kernel values of nodes in U_cp_v and U_nc_v
        states_cp = L[v,U_cp_v]
        states_nc = L[v,U_nc_v]
    
    # choose the node that edge detached from with probability proportional to 1/states_cp
    states_temp = 1/states_cp
    prob_cp = states_temp/np.sum(states_temp)
    j_minus = np.random.choice(U_cp_v,size=1,replace=False,p=prob_cp)
    # choose the node that edge reconnected to with probability proportional to states_nc
    if np.sum(states_nc)==0:
        states_nc = np.ones(len(states_nc))
    prob_nc = states_nc/np.sum(states_nc)
    j_add = np.random.choice(U_nc_v,size=1,replace=False,p=prob_nc)
    return j_add, j_minus