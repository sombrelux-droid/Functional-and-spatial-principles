# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:13:46 2020

@author: JiaLi
"""

import numpy as np
import sys
sys.path.append("..")
from scripts import basic_comp as bcomp

######################## deterministic adaptive rewiring #####################

def randRew(rand_adj_matrix, tau, p_in, p, rewirings):
    """
    The algorithm in Rentzeperis et al., 2022
    args:
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of random rewiring
        rewirings: number of rewirings to perform
    return:
        A: adjacency matrix
    """
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    
    for iter_i in range(rewirings):
        r_in = np.random.random_sample()
        if r_in<p_in: #rewire in-link
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)

            r = np.random.random_sample()
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)
            
            A[v, j_add] = A[v, j_minus]
            A[v, j_minus] = 0
        
        else: #rewire out-link
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)        
            
            A[j_add, v] = A[j_minus, v]
            A[j_minus, v] = 0
            
    return A

def distRew(vertices_coord, rand_adj_matrix, tau, p_in, p, rewirings):
    """
    Only use distance and adaptive rewiring principles.
    args:
        vertices_coord: positions of nodes on the unit dist
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of distance rewiring
        rewirings: number of rewirings to perform
    return:
        A: adjacency matrix
    """    
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    D = bcomp.compute_distance_matrix(vertices_coord)
    
    for iter_i in range(rewirings):
        r_in = np.random.random_sample()
        if r_in<p_in:
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)
            
            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)            
            A[v, j_add] = A[v, j_minus]
            A[v, j_minus] = 0
        
        else:
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)   
            A[j_add, v] = A[j_minus, v]
            A[j_minus, v] = 0
            
    return A
    
def tripRew(rand_adj_matrix, tau, p_in, p, q, rewirings, **kwargs):
    """
    At each iteration, the network is rewired according to one of 3 rewiring principles.
    args:
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of distance rewiring
        q: probability of wave rewiring
        rewirings: number of rewirings to perform
        **kwargs:
            vertices_coord: positions of nodes on the unit dist
            vector_field: an #nodes*2 array, containing the local vector field for each node
    return:
        A: adjacency matrix
    """
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    
    # settings of spatial rewiring principles
    if (p+q) > 0:
        for (key, value,) in kwargs.items():
            if key == "vertices_coord":
                vertices_coord = value
            elif key == "vector_field":
                vector_field = value
        D = bcomp.compute_distance_matrix(vertices_coord)
        if q>0:
            W_in = bcomp.compute_wave_matrix(vertices_coord, vector_field, D, flag='in')
            W_out = bcomp.compute_wave_matrix(vertices_coord, vector_field, D, flag='out')
    
    for iter_i in range(rewirings):
        r_in = np.random.random_sample()
        if r_in<p_in:
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)

            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            elif r<p+q: # wave rewiring
                j_add, j_minus = bcomp.wave_rewiring(v, U_cp_v, U_nc_v, W_in, flag)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            
            A[v, j_add] = A[v, j_minus]
            A[v, j_minus] = 0
        
        else:
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            elif r<p+q: # wave rewiring
                j_add, j_minus = bcomp.wave_rewiring(v, U_cp_v, U_nc_v, W_out, flag)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            A[j_add, v] = A[j_minus, v]
            A[j_minus, v] = 0
            
    return A

def randDistRew(vertices_coord, rand_adj_matrix, tau, p_in, p, q, rewirings):
    """
    Check if add a small proportion of random rewiring can improve average efficiency in distRew.
    args:
        vertices_coord: positions of nodes on the unit dist
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of random rewiring
        q: probability of distance rewiring
        rewirings: number of rewirings to perform
    return:
        A: adjacency matrix
    """    
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    D = bcomp.compute_distance_matrix(vertices_coord)
    
    for iter_i in range(rewirings):
        r_in = np.random.random_sample()
        if r_in<p_in:
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)
            
            r = np.random.random_sample()    
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            elif r<p+q: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)            
            A[v, j_add] = A[v, j_minus]
            A[v, j_minus] = 0
        
        else:
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()    
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            elif r<p+q: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus, flag_cp, flag_nc = bcomp.adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag)   
            A[j_add, v] = A[j_minus, v]
            A[j_minus, v] = 0
            
    return A

############# Stochastic version of adaptive rewiring #######################

def stochRandRew(rand_adj_matrix, tau, p_in, p, rewirings):
    """
    The algorithm in Rentzeperis et al., 2022
    args:
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of random rewiring
        rewirings: number of rewirings to perform
    return:
        A: adjacency matrix
    """
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    
    for iter_i in range(rewirings):
        flag_err = 0
        r_in = np.random.random_sample()
        if r_in<p_in: #rewire in-link
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)

            r = np.random.random_sample()
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            else: #adaptive rewiring
                j_add, j_minus = bcomp.stoch_adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            
            if flag_err == 0:
                A[v, j_add] = A[v, j_minus]
                A[v, j_minus] = 0
        
        else: #rewire out-link
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()
            if r<p: # random rewiring
                j_add, j_minus = bcomp.rand_rewiring(U_cp_v, U_nc_v)
            else: #adaptive rewiring
                j_add, j_minus = bcomp.stoch_adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            if flag_err == 0:
                A[j_add, v] = A[j_minus, v]
                A[j_minus, v] = 0
        if flag_err == 1:
            print('nega prob')
            return A
    return A

def stochDistRew(vertices_coord, rand_adj_matrix, tau, p_in, p, rewirings):
    """
    Only use distance and adaptive rewiring principles.
    args:
        vertices_coord: positions of nodes on the unit dist
        rand_adj_matrix: adjacency matrix of the initial random network
        tau: time constant of advection & consensus kernel
        p_in: probability of rewiring in-link in each iteration
        p: probability of distance rewiring
        rewirings: number of rewirings to perform
    return:
        A: adjacency matrix
    """    
    n = rand_adj_matrix.shape[0]
    A = rand_adj_matrix.copy()
    D = bcomp.compute_distance_matrix(vertices_coord)
    
    for iter_i in range(rewirings):
        flag_err = 0
        r_in = np.random.random_sample()
        if r_in<p_in:
            flag = 'in'
            deg_in = np.sum(A > 0, axis=1, keepdims=False,)
            nodes_receiving = np.where((deg_in > 0) & (deg_in < n- 1))[0]
            if len(nodes_receiving)==0:
                print("All nodes have either 0 or (n-1) in-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_receiving, flag)
            
            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus  = bcomp.stoch_adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            if flag_err == 0:
                A[v, j_add] = A[v, j_minus]
                A[v, j_minus] = 0
        
        else:
            flag = 'out'
            deg_out = np.sum(A > 0, axis=0, keepdims=False,)
            nodes_sending = np.where((deg_out > 0) & (deg_out < n - 1))[0]
            if len(nodes_sending)==0:
                print("All nodes have either 0 or (n-1) out-degree.")
                return A
        
            v, U_cp_v, U_nc_v = bcomp.choose_rewire_vertex(A, nodes_sending, flag)

            r = np.random.random_sample()    
            if r<p: # distance rewiring
                j_add, j_minus = bcomp.dist_rewiring(v, U_cp_v, U_nc_v, D)
            else: #adaptive rewiring
                j_add, j_minus  = bcomp.stoch_adapt_rewiring(v, U_cp_v, U_nc_v, A, tau, flag) 
            if flag_err == 0:
                A[j_add, v] = A[j_minus, v]
                A[j_minus, v] = 0
        if flag_err == 1:
            print('nega prob')
            return A
    return A
