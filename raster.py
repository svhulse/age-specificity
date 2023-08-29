import numpy as np
import pickle as pickle
import tqdm
import multiprocessing as mp

from model import Model

output_path = './data/raster.p'
size = 10

var_1 = 'mat'
var_2 = 'mu'

params = {  'b':1,				#Birthrate
			'mu':0.2,			#Deathrate
            'k':0.001,			#Coefficient of density dependent growth
            'mat':0.4}         #Maturation rate

mat_vals = np.linspace(0.3, 0.7, size)
mu_vals = np.linspace(0.05, 0.25, size)

vars = {'mat': mat_vals, 'mu': mu_vals}

def pass_to_sim(sim):
    return sim.run_sim()

if __name__ == '__main__':
	coords = []     #x, y coordinates of each simulation in raster
	sims = []     #Empty tuple for model classes

	#Create raster of model classes for each parameter combination
	for i in range(size):
		for j in range(size):
			coords.append((i,j))

			params[var_1] = vars[var_1][i]
			params[var_2] = vars[var_2][j]
			new_sim = Model(**params)

			sims.append(new_sim)
		
	#Run simluations for 4 core processor
	pool = mp.Pool(processes=2)	
	
	results = []
	for result in tqdm.tqdm(pool.imap(pass_to_sim, sims), total=len(sims)):
		results.append(result)

	raster = []
	for i in range(size):
		inds = [j for j in range(len(coords)) if coords[j][1] == i]
		raster.append([results[j] for j in inds])

	with open(output_path, 'wb') as f:
		pkl.dump([sims, results], f)