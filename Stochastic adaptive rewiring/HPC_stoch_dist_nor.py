import numpy as np
import pickle

import sys
sys.path.append("..")
from scripts import rewire

f = open('../Output/initials_nor.pckl', 'rb')
vertices_coord,initial_matrix = pickle.load(f)
f.close()

tau = 1
rewirings = 500
p_in_list = [0.5]
p_list = list(map(lambda x: x/10,range(11))) #p_distance

A_matrices = {}
for p_in in p_in_list:
    A_matrices[p_in] = {}
    for p in p_list:
        A_matrices[p_in][p] = {}
        for k in range(10):
            A_matrices[p_in][p][k] = {}
            A_matrices[p_in][p][k][0] = initial_matrix[k].copy()
            for i in range(30):
                A_matrices[p_in][p][k][rewirings*(i+1)] = rewire.stochDistRew(vertices_coord[k], A_matrices[p_in][p][k][rewirings*i], tau, p_in, p, rewirings)

f = open('../Output/stoch_dist_nor.pckl', 'wb')
pickle.dump(A_matrices, f)
f.close()
