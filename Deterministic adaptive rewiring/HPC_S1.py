import numpy as np
import pickle

import sys
sys.path.append("..")
from scripts import rewire

f = open('../Output/initials_nor.pckl', 'rb')
v_coord,initial_matrix = pickle.load(f)
f.close()

tau = 1
rewirings = 100
k = 0
n_vertices = 100

p_in_list = [0.0, 1.0]
vec_field = ['lat', 'rad']

pq_list = {} #key: p_distance, values: p_wave
pq_list[0.0] = [0.0,0.5,1.0]
pq_list[0.333] = [0.333]
pq_list[0.5] = [0.0]
pq_list[1.0] = [0.0]

A_matrices = {}
for p_in in p_in_list:
    A_matrices[p_in] = {}
    for p in pq_list.keys():
        for q in pq_list[p]:
            key = 'p='+str(p)+', q='+str(q)
            A_matrices[p_in][key] = {}
            if q==0.0:
                A_matrices[p_in][key][0] = initial_matrix[k].copy()
                for i in range(20):
                    A_matrices[p_in][key][rewirings*(i+1)] = rewire.tripRew(A_matrices[p_in][key][rewirings*i], tau, p_in, p, q, rewirings,vertices_coord=v_coord[k])
            else:
                for vf in vec_field:
                    A_matrices[p_in][key][vf] = {}
                    A_matrices[p_in][key][vf][0] = initial_matrix[k].copy()
                    if vf == 'lat':
                        vec_f = np.array([[1,0]]*n_vertices)
                    else:
                        vec_f = v_coord[k]
                    for i in range(20):
                        A_matrices[p_in][key][vf][rewirings*(i+1)] = rewire.tripRew(A_matrices[p_in][key][vf][rewirings*i], tau, p_in, p, q, rewirings,vertices_coord=v_coord[k],vector_field=vec_f)


f = open('../Output/S1.pckl', 'wb')
pickle.dump(A_matrices, f)
f.close()
