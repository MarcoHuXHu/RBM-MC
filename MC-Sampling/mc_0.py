import numpy as np

import math
import random

import spin_lattice as SL
    	
def MC_sampling():

	lat = SL.lattice(dim = 1, size = 4)
	
	n = lat.size ** lat.dim
	monte_carlo_steps = 10000;
	transient = 1000
	norm = (1.0 / float(monte_carlo_steps * n));
	E = 0; Esq = 0; Esq_avg = 0; E_avg = 0; etot = 0; etotsq = 0
	M = 0; Msq = 0; Msq_avg = 0; M_avg = 0; mtot = 0; mtotsq = 0
	Mabs = 0; Mabs_avg = 0; mabstot = 0; mqtot = 0
	T = 2.0; minT = 0.5; deltT = 0.1
	

	for s in lat.config:
		print(s.value())
	
		
#     ft = open('temperature.txt', 'w')
#     feavg = open('E_avg.txt', 'w'); fesqavg = open('Esq_avg.txt', 'w')
#     fmavg = open('M_avg.txt', 'w'); fmsqavg = open('Msq_avg.txt', 'w')
#     fmabsavg = open('Mabs_avg.txt', 'w')
# 
#     while T >= minT:
#         # Transient Function
#         for _ in range(transient):
#             for _ in range(n):               # flip all spins
#                 lat.metropolis(T)
#         M = lat.magnetization()
#         Mabs = abs(M)
#         E = lat.energy()
# 
#         # Initialize summation variables at each temperature step
#         etot = 0; etotsq = 0; mtot = 0; mtotsq = 0; mabstot = 0; mqtot = 0
# 
#         # Monte Carlo loop
#         for _ in range(monte_carlo_steps):
#             for _ in range(n):
#                 de, pos, flip = lat.metropolis(T)
#                 if flip:  
#                     E += 2 * de    # Factor 2 because we want to count E on all site, eventually would / 2
#                     v  = lat.get_spin_by_lattice_position(pos).value()
#                     M += 2 * v
#                     Mabs += abs(2 * v)
# 
#             # Keep summation of observables
#             etot += E / 2.0   # so as not to count the energy for each spin twice
#             etotsq += E / 2.0 * E / 2.0
#             mtot += M
#             mtotsq += M * M
#             mqtot += M * M * M * M
#             mabstot += (math.sqrt(M * M))
# 
#         # Average observables
#         E_avg = etot * norm; Esq_avg = etotsq * norm;
#         M_avg = mtot * norm; Msq_avg = mtotsq * norm; Mabs_avg = mabstot * norm; Mq_avg = mqtot * norm;
# 
#         # output
#         st = "{:.9f}".format(T) + '\n'; ft.write(st)
#         seavg = "{:.9f}".format(E_avg) + '\n'; feavg.write(seavg);
#         sesqavg = "{:.9f}".format(Esq_avg) + '\n'; fesqavg.write(sesqavg)
#         smavg = "{:.9f}".format(M_avg) + '\n'; fmavg.write(smavg);
#         smsqavg = "{:.9f}".format(Msq_avg) + '\n'; fmsqavg.write(smsqavg)
#         smabsavg = "{:.9f}".format(Mabs_avg) + '\n'; fmabsavg.write(smabsavg)
# 
#         print("{:.3f}".format(T))
#         T -= deltT
# 
#     print("Finished")
#     ft.close()
#     feavg.close(); fesqavg.close()
#     fmavg.close(); fmsqavg.close()
#     fmabsavg.close()


if __name__ == '__main__':
    MC_sampling()
