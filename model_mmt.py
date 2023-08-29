import numpy as np

from scipy.integrate import solve_ivp

__author__ = 'Samuel Hulse'

class ModelMMT:
	'''
	The Model class is used to define a simulation for the QR host-pathogen
	model. It allows for both density-dependent and frequency-dependent disease
	transmission. Parameters can also be changed between multiple runs by 
	using kwargs in the run_sim method.
	'''

	def __init__(self, **kwargs):
		self.N_alleles = 100 #number of alleles
		self.N_iter = 60 #number of mutational time steps
		mut = 0.05 #mutation rate

		#Define the mutation matrix
		self._M = np.diag(np.full(self.N_alleles, 1 - mut))
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, 1)
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, -1)
		self._M[0,1] = mut
		self._M[self.N_alleles - 1, self.N_alleles - 2] = mut

		#Set parameters and resistance-cost curve
		self.beta_j = np.linspace(0, 0.005, self.N_alleles)
		self.beta_a = np.linspace(0.5, 0, self.N_alleles)

		#Set default parameters
		self.b = 1
		self.mu = 0.2
		self.gamma = 0.01
		self.mat = 0.4

		#Get kwargs from model initialization and modify parameter values, if
		#h is changed, then _t and N_t must also be changed to compensate
		for key, value in kwargs.items():
			setattr(self, key, value)

	#Differential equation for frequency-dependent transmission
	def df(self, t, X):
		J = X[0]
		A = X[1]
		I = X[2:]

		N = A + J + np.sum(I)
		
		dJ = A*(self.b) - J*(self.mat + self.gamma*N + self.mu + np.dot(self.beta_j, I))
		dA = J*(self.mat) - A*(self.gamma*N + self.mu + np.dot(self.beta_a, I) / N)
		dI = J*np.multiply(self.beta_j, I) + A*np.multiply(self.beta_a, I)/N - I*self.mu

		return np.concatenate([[dJ, dA], dI])

	#Run simulation
	def run_sim(self, t=(0, 5000)):
		#Set initial conditions
		J = np.zeros(self.N_iter)
		A = np.zeros(self.N_iter)
		I = np.zeros((self.N_alleles, self.N_iter))
		
		J[0] = 1
		A[0] = 1
		I[50, 0] = 1
		zero_threshold = 0.01 #Threshold to set abundance values to zero

		for i in range(self.N_iter - 1):
			X_0 = np.concatenate(([J[i], A[i]], I[:,i]))
				
			sol = solve_ivp(self.df, t, X_0)

			J[i+1] = sol.y[0, -1]
			A[i+1] = sol.y[1, -1]
			I[:,i+1] = np.dot(self._M, sol.y[2:, -1])

			#Set any population below threshold to 0
			for j in range(self.N_alleles):
				if I[j, -1] < zero_threshold:
					I[j, -1] = 0
				if I[j, -1] < zero_threshold:
					I[j, -1] = 0

		return (J, A, I)