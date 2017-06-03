from __future__ import print_function

import timeit
import pickle

import numpy as np

import theano
import theano.tensor as T

from mc_ising import MC_sampling
from RBM import rbm_training


data_name = 'tau'

T_range = [0.5, 3.0]
T_step = 1.5
N_tau = int((T_range[1] - T_range[0]) // T_step)

dim = 1
size = 8


def RBM_for_Ising():

	MC_sampling(data_name = data_name, 
				T_range = T_range,
				dT = T_step,
				dim = dim,
				size = size
				)
	
	Temperature = []
	Weight = []; visible_bias = []; hidden_bias = []
	
	for tau in range(N_tau):
		file_name = data_name + str(T_range[1] - tau*T_step) + '.dat'
		w, vb, hb = rbm_training(dataset = file_name, n_hidden = 6)
		
		Temperature.append(T_range[1] - tau*T_step)
		Weight.append(w); visible_bias.append(vb); hidden_bias.append(hb)
		
# 	print(Weight[0].eval())
		
	

		
if __name__ == '__main__':
    RBM_for_Ising()
