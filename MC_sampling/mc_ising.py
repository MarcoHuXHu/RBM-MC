import math
import random

import pickle
import timeit

from ising_lattice import Spin
from ising_lattice import lattice

    	
def MC_sampling(data_name = 'tau',
				T_range = [0.5, 4.5],
				dT = 1.0,
				dim = 1,
				size = 8):
	"""
	Monte Carlo sampling for an Ising model on a hyper-square lattice;
	Output: thermaldynamic observables: Energy, Magnetization
			spin configs samples
	
	:param  data_name: name of the file storing spin configs at different temperatures
	
	:param  T_range: list object, restrict temperature range for MC sampling
	
	:param  dT: temperature step for imagnary-time evolution
	
	:param  dim: dimension of lattice
	
	:param  size: size of the lattice
	
	"""
	feavg = open('E_avg.txt','w'); fesqavg = open('Esq_avg.txt','w');
	fmavg = open('M_avg.txt', 'w'); fmsqavg = open('Msq_avg.txt', 'w')
	fmabsavg = open('Mabs_avg.txt', 'w')

	lat = lattice(dim = dim, size = size)
	
	n = lat.size ** lat.dim
	monte_carlo_steps = 100; transient = 1000
	norm = (1.0 / float(monte_carlo_steps * n));
	T = T_range[1]; minT = T_range[0]; deltT = dT
	
	E = 0; Esq = 0; Esq_avg = 0; E_avg = 0; etot = 0; etotsq = 0
	M = 0; Msq = 0; Msq_avg = 0; M_avg = 0; mtot = 0; mtotsq = 0
	Mabs = 0; Mabs_avg = 0; mabstot = 0; mqtot = 0
	
	start_time = timeit.default_timer()
	
	while T >= minT:				
		for _ in range(transient):
			for _ in range(n):               # flip all spins
				lat.metropolis(T)
		M = lat.magnetization()
		Mabs = abs(M)
		E = lat.energy()
		
		# Initialize sum variables at each temperature
		etot = 0; etotsq = 0; mtot = 0; mtotsq = 0; mabstot = 0; mqtot = 0
		data = []
		
		# Monte Carlo loop
		for _ in range(monte_carlo_steps):
			for _ in range(n):
				de, pos, flip = lat.metropolis(T)
				if flip:
					E += 2 * de    # Factor 2 because would / 2
					v  = lat.get_spin_by_lattice_position(pos).value()
					M += 2 * v
					Mabs += abs(2 * v)
			states, values = lat.get_output()
			data.append(values)
			
			# sum of observables
			etot += E / 2.0 
			etotsq += E / 2.0 * E / 2.0
			mtot += M
			mtotsq += M * M
			mqtot += M * M * M * M
			mabstot += (math.sqrt(M * M))
			
		# Average observables
		E_avg = etot * norm; Esq_avg = etotsq * norm;
		M_avg = mtot * norm; Msq_avg = mtotsq * norm; 
		Mabs_avg = mabstot * norm; Mq_avg = mqtot * norm;
		
		# output
		seavg = "{:.9f}".format(E_avg) + '\n'; feavg.write(seavg);
		sesqavg = "{:.9f}".format(Esq_avg) + '\n'; fesqavg.write(sesqavg)
		smavg = "{:.9f}".format(M_avg) + '\n'; fmavg.write(smavg);
		smsqavg = "{:.9f}".format(Msq_avg) + '\n'; fmsqavg.write(smsqavg)
		smabsavg = "{:.9f}".format(Mabs_avg) + '\n'; fmabsavg.write(smabsavg)
		
		# create training data for RBM
		with open(data_name + str(T) + ".dat", 'wb') as data_file:
			pickle.dump(data, data_file)
		data_file.close()
		
# 		print('Finished sampling at temperature %.3f' % T)
		T -= deltT
		
	end_time = timeit.default_timer()
	pretraining_time = (end_time - start_time) 
	
	print("Monte-Carlo sampling Finished")
	print ('Sampling took %f minutes' % (pretraining_time / 60.))
	
	feavg.close(); fesqavg.close()
	fmavg.close(); fmsqavg.close()
	fmabsavg.close()

