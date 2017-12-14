import walkerSimEnv as sim
import numpy as np 
import pickle

ITERMAX = 100
l = 100*100*10 

my_sim = sim.walkerSimulation()
totalTimes = []
for epoc in range(5):
	print('epoc '+str(epoc)+' is running...')
	# run first round of simulation
	X_0 = [1.52]
	Y_0 = [0.36]
	trjs = my_sim.run([X_0, Y_0], nstepmax = l) # 2xl
	if  np.any(np.array(trjs[1])>1.5): # terminate condition
			totalTime = np.where(np.array(trjs[1])>1.5)[0][0]
			print('The simulation reached the final point in '+str(totalTime)+' steps')
	totalTimes.append(totalTime)
	
np.savetxt('totalTimes', totalTimes)
