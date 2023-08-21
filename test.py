def df(self, t, X):
	J = X[0]
	A = X[1]
	I = X[2:]

	N = A + J + np.sum(I)
		
	dJ = A*self.b - self.gamma*N - J*(self.mat + self.mu + np.dot(self.beta_j, I)/N)
	dA = J*self.mat - A*(self.mu + np.dot(self.beta_a, I)/N)
	dI = J*np.dot(self.beta_j, I)/N + A*np.dot(self.beta_a, I)/N - I*self.mu

	return np.concatenate([[dJ, dA], dI])