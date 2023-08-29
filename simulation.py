from model import Model

from matplotlib import pyplot as plt

params = {  'b':1,				#Birthrate
			'mu':0.2,			#Deathrate
            'k':0.001,			#Coefficient of density dependent growth
            'mat':0.3}          #Maturation rate

sim = Model(**params)
J, A, I = sim.run_sim()

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].imshow(I)
ax[0].set_ylabel('Specialization')
ax[0].set_xlabel('Time')

ax[1].plot(J/(J+A))
ax[1].set_ylabel('Proportion Juvenile')
ax[1].set_xlabel('Time')