import walkerSimEnv as sim
import numpy as np 

X_0 = [1.52]
Y_0 = [0.36]

N = len(X_0) # number of parallel runs 
l = 10
# run first round of simulation
my_sim = sim.walkerSimulation()
trjs = my_sim.run([X_0, Y_0], nstepmax = l)
i = my_sim.updateState(trjs)

for round in range(700):
	# find a value for control 
	u = my_sim.findU(i, method = 'QL')
	newPoints = my_sim.findStarting(trjs, u, method = 'QL')

	trj1 = my_sim.run(newPoints, nstepmax = l)
	com_trjs = []
	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta]))) #check dim
	trjs = np.array(com_trjs)
	# update the state of the system
	j = my_sim.updateState(trjs) # based on all trajectories
	
	# calculate imidiate cost
	cost = my_sim.cost(i, j)
	
	# update Q function
	my_sim.updateQ(i, j, u, cost) # important
	i = j
	
	if finalState:
		halt
		
np.save('trjs', trjs)
