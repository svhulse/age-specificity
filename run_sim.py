import numpy as np
from scipy.integrate import solve_ivp

from model import Model

N_alleles = 40

beta_j = np.linspace(0.3, 0.7, N_alleles)
beta_a = np.linspace(0.7, 0.3, N_alleles)

b = 1
mu = 0.2
gamma = 0.01
mat = 0.5

mut = 0.05 #mutation rate

#Define the mutation matrix
M = np.diag(np.full(N_alleles, 1 - mut))
M = M + np.diag(np.ones(N_alleles - 1)*mut/2, 1)
M = M + np.diag(np.ones(N_alleles - 1)*mut/2, -1)
M[0,1] = mut
M[N_alleles - 1, N_alleles - 2] = mut

def df(t, X):
	J = X[0]
	A = X[1]
	I = X[2:]

	N = np.sum(X)
		
	dJ = A*(b - gamma*N) - J*(mat + mu + np.dot(beta_j, I)/N)
	dA = J*(mat - gamma*N) - A*(mu + np.dot(beta_a, I)/N)
	dI = J*np.multiply(beta_j, I)/N + A*np.multiply(beta_a, I)/N - I*mu

	return np.concatenate([[dJ, dA], dI])

def run_sim(N_iter, t=(0, 1000)):
	#Set initial conditions
	J = np.zeros(N_iter)
	A = np.zeros(N_iter)
	I = np.zeros((N_alleles, N_iter))
		
	J[0] = 1
	A[0] = 1
	I[20, 0] = 1
	zero_threshold = 0.01 #Threshold to set abundance values to zero

	for i in range(N_iter - 1):
		print(i)
		X_0 = np.concatenate([[J[i], A[i]], I[:,i]])
				
		sol = solve_ivp(df, t, X_0, method='DOP853')

		J[i+1] = sol.y[0,-1]
		A[i+1] = sol.y[1,-1]
		I[:,i+1] = np.dot(M, sol.y[2:,-1])

		#Set any population below threshold to 0
		for j in range(N_alleles):
			if I[j, -1] < zero_threshold:
				I[j, -1] = 0
			if I[j, -1] < zero_threshold:
				I[j, -1] = 0

	return (J, A, I)

sim = Model()

N_iter = 50