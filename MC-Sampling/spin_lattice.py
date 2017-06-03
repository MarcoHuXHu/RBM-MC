import numpy as np

import math
import random


class Spin(object):
    def __init__(self, 
    			 mag = 0.5, 
    			 state = None,
    			 dic = None
    			 ):
    	dic = {}
    	for i in range(int(mag*2+1)):
#     		dic[i] = i - mag
    		dic[i] = int(i*2 - mag*2)
    	self.mag = mag
    	self.dic = dic
    	if state:
    		self.state = state
    	else:
    		self.randomize()
    	
    def flip(self):
        self.state = 1 - self.state
        
    def randomize(self):
    	self.state = random.randint(0, len(self.dic)-1)
    	
    def value(self):
    	return self.dic[self.state]
    	
    def copy(self):
    	s = Spin(state = self.state)
    	return s



class lattice(object):
	def __init__(self, dim = 3, size = 4, config = None):
		self.dim = dim
		self.size = size
		if config:
			self.config = config
		else:
			self.initialization()
			
	def initialization(self):
		if self.dim == 1:
			fig_1 = []
			for _ in range(self.size):
				fig_1.append(Spin())
			self.config = fig_1
		else:
			fig_d = []
			for _ in range(self.size):
				fig_d.append(lattice(dim=(self.dim-1),
									 size=self.size
									).config)
			self.config = fig_d
    				
	def get_output(self):
		result = self.config
		for _ in range(self.dim - 1):
			result = sum(result, [])
		return [spin.state for spin in result]
    	
	def nearby(self, pos):  
		nears = []
		for i in range(self.dim):
			left = list(pos)
			right= list(pos)
			left[i] -= 1
			right[i]+= 1
			if left[i] < 0:
				left[i] = self.size - 1
			if right[i] == self.size:
				right[i] = 0
			nears.append(left)
			nears.append(right)
		return nears 
    	
	def get_random_position(self):       # returns a position vector pos[x,y,z]
		pos = []
		for _ in range(self.dim):
			pos.append(random.randint(0, self.size - 1))
		return pos
    	
	def get_spin_by_lattice_position(self, pos):  # returns spin at site = pos
		getter = self.config
		for i in range(self.dim):
			getter = getter[pos[i]]
		return getter
    	
	def site_energy(self, pos):           # returns energy at site = pos
		nears = self.nearby(pos)
		energy = 0
		for i in range(len(nears)):
			energy = energy + self.get_spin_by_lattice_position(nears[i]).value()
		energy *= -1 * self.get_spin_by_lattice_position(pos).value()
		return energy
    	
	def metropolis(self, T):                      # find a random spin and flip or not
		pos = self.get_random_position()
		de = -2 * self.site_energy(pos)
		if (de < 0) or (random.random() < math.exp(-de/T)):
			self.get_spin_by_lattice_position(pos).flip()
			return de, pos, True
		return de, pos, False
    	
	def all_sites(self):
		res = []
		def traverse(pos):
			for x in range(self.size):
				pos.append(x)
				if len(pos) == self.dim:
					res.append(list(pos))
				else:
					traverse(pos)
				pos.pop()
		traverse([])
		return res
    	
	def energy(self):
		e = 0
		position = self.all_sites()
		for pos in position:
			e += self.site_energy(pos)
		return e
    	
	def magnetization(self):
		m = 0 
		position = self.all_sites()
		for pos in position:
			m += self.get_spin_by_lattice_position(pos).value()
		return m
