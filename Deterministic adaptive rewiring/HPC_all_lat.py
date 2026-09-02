import numpy as np
import pickle

import sys
sys.path.append("..")
from scripts import rewire

f = open('../Output/initials_nor.pckl', 'rb')
v_coord,initial_matrix = pickle.load(f)
f.close()

n_vertices = 100
p_in = 0.5
tau = 1
rewirings = 500
p_list = list(map(lambda x: x/10,range(11))) #p_distance
q_list = [0.1, 0.3, 0.5]
vec_field =  np.array([[1,0]]*n_vertices)

A_matrices = {}
for q in q_list:
    A_matrices[q] = {}
    for p in p_list:
        if (p+q)<=1:
            A_matrices[q][p] = {}
            for k in range(10):
                A_matrices[q][p][k] = {}
                A_matrices[q][p][k][0] = initial_matrix[k]
                for i in range(30):
                    A_matrices[q][p][k][rewirings*(i+1)] = rewire.tripRew(A_matrices[q][p][k][rewirings*i], tau, p_in, p, q, rewirings, 
                                      vertices_coord = v_coord[k], vector_field = vec_field)

f = open('../Output/all_lat_nor.pckl', 'wb')
pickle.dump(A_matrices, f)
f.close()
