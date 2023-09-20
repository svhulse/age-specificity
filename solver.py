import numpy as np
import sympy as sym

from sympy.solvers import solve
from sympy import Symbol

J = Symbol('J')
A = Symbol('A')
I = Symbol('I')

b, mat, gamma, mu = sym.symbols('b mat gamma mu')
beta_j, beta_a = sym.symbols('beta_j beta_a')
		
eq1 = A*(b) - J*(mat + gamma*(J+A+I) + mu + beta_j*I)
eq2 = J*(mat) - A*(mu + beta_a*I)
eq3 = I*(beta_j*J + beta_a*A - mu)

result = sym.solve([eq1, eq2, eq3], (J,A,I))
print(result)

eq1 = A*(b) - J*(mat + gamma*(J+A) + mu)
eq2 = J*(mat) - A*(mu)

result = sym.solve([eq1, eq2, eq3], J)