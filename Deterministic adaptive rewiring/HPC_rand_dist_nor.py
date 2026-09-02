import numpy as np
import pickle

import sys
sys.path.append("..")
from scripts import rewire

f = open('../Output/initials_nor.pckl', 'rb')
vertices_coord,initial_matrix = pickle.load(f)
f.close()

p_in = 0.5
tau = 1
rewirings = 1000

p = 0.1 # p_random
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #p_distance

A_matrices = {}
for q in q_list:
    A_matrices[q] = {}
    for k in range(10):
        A_matrices[q][k] = {}
        A_matrices[q][k][0] = initial_matrix[k].copy()
        for i in range(15):
            A_matrices[q][k][rewirings*(i+1)] = rewire.randDistRew(vertices_coord[k], A_matrices[q][k][rewirings*i], tau, p_in, p, q, rewirings)

f = open('../Output/rand_dist_nor.pckl', 'wb')
pickle.dump(A_matrices, f)
f.close()
